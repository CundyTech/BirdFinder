import { createApi, fetchBaseQuery, retry } from '@reduxjs/toolkit/query/react';
import { getCachedSpeciesInfo, setCachedSpeciesInfo } from './birdInfoCache';
import { getCachedRarityMap, setCachedRarityMap } from './ukRarityCache';
import { createSemaphore } from './asyncSemaphore';
import { SPECIES } from '../species';

// Free iNaturalist API, no key required.
const TAXA_BASE_URL = 'https://api.inaturalist.org/v1/taxa';
const OBSERVATIONS_URL = 'https://api.inaturalist.org/v1/observations';
// iNaturalist place id for the United Kingdom (admin_level 0) — this app is
// UK-only, so rarity should reflect how often a species turns up here, not
// its global observation count.
const UK_PLACE_ID = 6857;
const REQUEST_TIMEOUT_MS = 8000;
const SUMMARY_MAX_LENGTH = 220;
const MAX_PLACES_PER_GROUP = 10;
const MAX_GALLERY_PHOTOS = 4;
// Rough "how settled is it there" ordering so native/endemic reads before introduced.
const MEANS_PRIORITY = ['endemic', 'native', 'introduced', 'naturalised', 'invasive'];

// The life-list grid can mount 60 tiles at once, each wanting a lookup —
// cap how many of those run concurrently instead of firing them all at the
// free, unauthenticated API simultaneously.
const MAX_CONCURRENT_LOOKUPS = 4;
const runThrottled = createSemaphore(MAX_CONCURRENT_LOOKUPS);

function stripHtml(html) {
  if (!html) return '';
  return html.replace(/<[^>]*>/g, '').trim();
}

function truncate(text, maxLength) {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength).replace(/\s+\S*$/, '') + '…';
}

// Countries the species is known from, grouped by native/introduced/etc,
// from iNaturalist's curated regional checklists (not a heatmap of user
// observations, which would skew toward wherever birders happen to be).
function summarizeRange(listedTaxa) {
  if (!Array.isArray(listedTaxa) || listedTaxa.length === 0) return null;

  const countries = listedTaxa.filter(
    (entry) => entry.place && entry.place.admin_level === 0 && entry.establishment_means
  );
  if (countries.length === 0) return null;

  const byMeans = new Map();
  for (const entry of countries) {
    const means = entry.establishment_means;
    if (!byMeans.has(means)) byMeans.set(means, new Set());
    byMeans.get(means).add(entry.place.name);
  }

  const groups = Array.from(byMeans.entries()).map(([means, placeSet]) => {
    const places = Array.from(placeSet).sort();
    const shown = places.slice(0, MAX_PLACES_PER_GROUP);
    const overflow = places.length - shown.length;
    return {
      meansLabel: means.charAt(0).toUpperCase() + means.slice(1).replace(/_/g, ' '),
      placesText: overflow > 0 ? `${shown.join(', ')} +${overflow} more` : shown.join(', '),
      priority: MEANS_PRIORITY.indexOf(means),
    };
  });

  groups.sort((a, b) => (a.priority === -1 ? 99 : a.priority) - (b.priority === -1 ? 99 : b.priority));
  return groups.map(({ meansLabel, placesText }) => ({ meansLabel, placesText }));
}

// Retries transient failures (timeouts, 5xx, rate-limit 429s) with
// exponential backoff — the throttle above reduces how often those happen,
// this recovers when one slips through anyway.
const baseQueryWithRetry = retry(fetchBaseQuery({ baseUrl: TAXA_BASE_URL, timeout: REQUEST_TIMEOUT_MS }), {
  maxRetries: 3,
});

export const birdInfoApi = createApi({
  reducerPath: 'birdInfoApi',
  baseQuery: baseQueryWithRetry,
  endpoints: (builder) => ({
    getSpeciesInfo: builder.query({
      // The search endpoint gives the photo/status/scientific name, but
      // ancestor taxonomy (family/order) and the Wikipedia summary only
      // come back from the single-taxon endpoint — queryFn lets us sequence
      // both calls through the same baseQuery for one cache entry.
      async queryFn(commonName, _queryApi, _extraOptions, baseQuery) {
        // Species facts barely change — check the on-disk cache before
        // touching the network (or the throttle queue) at all, so cache
        // hits return instantly instead of waiting behind live lookups.
        const cached = await getCachedSpeciesInfo(commonName);
        if (cached !== undefined) return { data: cached };

        return runThrottled(() => fetchSpeciesInfo(commonName, baseQuery));
      },
      // Species facts don't change — once fetched, never treat as stale.
      keepUnusedDataFor: Infinity,
    }),

    // One bulk request covering the whole 60-species roster, instead of
    // every life-list tile making its own UK-observations lookup — that
    // was firing 60 requests at once on mount and reliably tripping
    // timeouts/rate limits, so most of them came back empty.
    getUkRarityMap: builder.query({
      async queryFn(_arg, _queryApi, _extraOptions, baseQuery) {
        const cached = await getCachedRarityMap();
        if (cached !== undefined) return { data: cached };

        const taxonParams = SPECIES.map((s) => `taxon_id[]=${s.taxonId}`).join('&');
        const result = await baseQuery(
          `${OBSERVATIONS_URL}/species_counts?place_id=${UK_PLACE_ID}&verifiable=true&per_page=${SPECIES.length}&${taxonParams}`
        );
        if (result.error) return { error: result.error };

        const countByTaxonId = new Map((result.data?.results || []).map((r) => [r.taxon?.id, r.count]));
        const map = {};
        for (const species of SPECIES) {
          const count = countByTaxonId.get(species.taxonId);
          map[species.id] = typeof count === 'number' ? count : 0;
        }

        await setCachedRarityMap(map);
        return { data: map };
      },
      keepUnusedDataFor: Infinity,
    }),
  }),
});

async function fetchSpeciesInfo(commonName, baseQuery) {
  try {
    const searchResult = await baseQuery(
      `?q=${encodeURIComponent(commonName)}&rank=species&iconic_taxa[]=Aves&per_page=1`
    );
    // A real request failure (network/timeout/5xx) is retryable —
    // surface it as an error. "No match found" is not a failure.
    if (searchResult.error) return { error: searchResult.error };

    const match = searchResult.data?.results?.[0];
    if (!match) return { data: null };

    let ancestors = [];
    let summary = '';
    let wikipediaUrl = null;
    let range = null;
    let galleryPhotos = [];
    const detailResult = await baseQuery(`/${match.id}`);
    if (!detailResult.error) {
      const detail = detailResult.data?.results?.[0];
      ancestors = detail?.ancestors || [];
      summary = truncate(stripHtml(detail?.wikipedia_summary), SUMMARY_MAX_LENGTH);
      wikipediaUrl = detail?.wikipedia_url || null;
      range = summarizeRange(detail?.listed_taxa);
      galleryPhotos = (detail?.taxon_photos || [])
        .map((tp) => tp.photo?.medium_url)
        .filter((url) => url && url !== match.default_photo?.medium_url)
        .slice(0, MAX_GALLERY_PHOTOS);
    }

    const family = ancestors.find((a) => a.rank === 'family');
    const order = ancestors.find((a) => a.rank === 'order');

    const data = {
      commonName: match.preferred_common_name || commonName,
      scientificName: match.name || null,
      photoUrl: match.default_photo?.medium_url || null,
      photoAttribution: match.default_photo?.attribution || null,
      conservationStatus: match.conservation_status?.status_name || null,
      family: family?.name || null,
      order: order?.name || null,
      // Global popularity on iNaturalist, not UK-specific — the rarity
      // meter uses getUkRarityMap's bulk, UK-scoped counts instead.
      observationsCount: typeof match.observations_count === 'number' ? match.observations_count : null,
      summary,
      wikipediaUrl,
      range,
      galleryPhotos,
    };
    await setCachedSpeciesInfo(commonName, data);
    return { data };
  } catch (err) {
    return { error: { status: 'FETCH_ERROR', error: err?.message || 'Unknown error' } };
  }
}

export const { useGetSpeciesInfoQuery, useGetUkRarityMapQuery } = birdInfoApi;
