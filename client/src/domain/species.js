import i18n from '../i18n';

// The 60 species the model was trained on (model/h5/labels.json), duplicated
// here so the client has its own copy to build the life list roster from —
// keep in sync if the model is retrained with a different species set.
export const SPECIES_IDS = [
  'Barn_Owl',
  'Barn_Swallow',
  'Black_headed_Gull',
  'Canada_Goose',
  'Carrion_Crow',
  'Coal_Tit',
  'Common_Blackcap',
  'Common_Buzzard',
  'Common_Chiffchaff',
  'Common_Kestrel',
  'Common_Kingfisher',
  'Common_Linnet',
  'Common_Moorhen',
  'Common_Pheasant',
  'Common_Raven',
  'Common_Swift',
  'Dunnock',
  'Eurasian_Blackbird',
  'Eurasian_Blue_Tit',
  'Eurasian_Bullfinch',
  'Eurasian_Chaffinch',
  'Eurasian_Collared_Dove',
  'Eurasian_Coot',
  'Eurasian_Greenfinch',
  'Eurasian_Jackdaw',
  'Eurasian_Jay',
  'Eurasian_Magpie',
  'Eurasian_Nuthatch',
  'Eurasian_Oystercatcher',
  'Eurasian_Skylark',
  'Eurasian_Sparrowhawk',
  'Eurasian_Treecreeper',
  'Eurasian_Wren',
  'European_Goldfinch',
  'European_Herring_Gull',
  'European_Robin',
  'European_Starling',
  'Great_Spotted_Woodpecker',
  'Great_Tit',
  'Green_Woodpecker',
  'Grey_Heron',
  'Grey_Partridge',
  'Grey_Wagtail',
  'House_Martin',
  'House_Sparrow',
  'Long_tailed_Tit',
  'Mallard',
  'Mistle_Thrush',
  'Mute_Swan',
  'Northern_Lapwing',
  'Pied_Wagtail',
  'Red_Kite',
  'Reed_Bunting',
  'Rook',
  'Song_Thrush',
  'Stock_Dove',
  'Tawny_Owl',
  'Willow_Warbler',
  'Woodpigeon',
  'Yellowhammer',
];

// iNaturalist taxon id for each species, resolved once against
// /v1/observations/species_counts (place_id=6857, i.e. the UK) and pinned
// here so the rarity meter can fetch all 60 counts in a single bulk
// request instead of one lookup per species. A few of these differ from
// our label because iNaturalist's preferred common name doesn't match the
// UK/trained-model name for the same species (e.g. Common Pheasant is
// listed there as "Ring-necked Pheasant"; Pied Wagtail, the UK subspecies,
// rolls up under the species-level "White Wagtail").
export const TAXON_IDS = {
  Barn_Owl: 20445,
  Barn_Swallow: 11901,
  Black_headed_Gull: 144510,
  Canada_Goose: 7089,
  Carrion_Crow: 204496,
  Coal_Tit: 144823,
  Common_Blackcap: 15282,
  Common_Buzzard: 204472,
  Common_Chiffchaff: 117016,
  Common_Kestrel: 472766,
  Common_Kingfisher: 2599,
  Common_Linnet: 558634,
  Common_Moorhen: 201282,
  Common_Pheasant: 981,
  Common_Raven: 8010,
  Common_Swift: 6638,
  Dunnock: 13988,
  Eurasian_Blackbird: 12716,
  Eurasian_Blue_Tit: 144849,
  Eurasian_Bullfinch: 9462,
  Eurasian_Chaffinch: 10070,
  Eurasian_Collared_Dove: 2969,
  Eurasian_Coot: 482,
  Eurasian_Greenfinch: 145360,
  Eurasian_Jackdaw: 336399,
  Eurasian_Jay: 8088,
  Eurasian_Magpie: 891696,
  Eurasian_Nuthatch: 14824,
  Eurasian_Oystercatcher: 4843,
  Eurasian_Skylark: 7347,
  Eurasian_Sparrowhawk: 5106,
  Eurasian_Treecreeper: 7450,
  Eurasian_Wren: 145363,
  European_Goldfinch: 9398,
  European_Herring_Gull: 204533,
  European_Robin: 13094,
  European_Starling: 14850,
  Great_Spotted_Woodpecker: 17871,
  Great_Tit: 203153,
  Green_Woodpecker: 144243,
  Grey_Heron: 4954,
  Grey_Partridge: 831,
  Grey_Wagtail: 13688,
  House_Martin: 64705,
  House_Sparrow: 13858,
  Long_tailed_Tit: 7278,
  Mallard: 6930,
  Mistle_Thrush: 12735,
  Mute_Swan: 6921,
  Northern_Lapwing: 4857,
  Pied_Wagtail: 13695,
  Red_Kite: 5267,
  Reed_Bunting: 9201,
  Rook: 8029,
  Song_Thrush: 12748,
  Stock_Dove: 3033,
  Tawny_Owl: 19898,
  Willow_Warbler: 117055,
  Woodpigeon: 3048,
  Yellowhammer: 9195,
};

// "Bird family" trophy groupings — informal, common-name-based types a
// birdwatcher would recognise (e.g. "the tits"), not strict taxonomic
// families. That's deliberate: Long-tailed Tit is scientifically a
// different family from the other tits (Aegithalidae vs Paridae), but
// splitting it out would be a confusing trophy for anyone but a taxonomist.
// Every one of the 60 species belongs to exactly one group.
// Each group's `key` is the frozen English label used as its trophy-storage
// identity (see useTrophyCategories.js's makeTrophy) — it must never change
// even if `label`'s translation does, since it's part of the persisted
// claimed-trophy key. `label` is what's actually displayed, resolved
// through i18n so a future language just adds translations for these keys.
export const TYPE_GROUPS = [
  { id: 'tits', key: 'Tits', label: i18n.t('groups.type.Tits') },
  { id: 'corvids', key: 'Crows & Corvids', label: i18n.t('groups.type.Crows & Corvids') },
  { id: 'raptors', key: 'Birds of Prey', label: i18n.t('groups.type.Birds of Prey') },
  { id: 'owls', key: 'Owls', label: i18n.t('groups.type.Owls') },
  { id: 'waterfowl', key: 'Waterfowl', label: i18n.t('groups.type.Waterfowl') },
  { id: 'gulls', key: 'Gulls', label: i18n.t('groups.type.Gulls') },
  { id: 'thrushes', key: 'Thrushes', label: i18n.t('groups.type.Thrushes') },
  { id: 'finches', key: 'Finches', label: i18n.t('groups.type.Finches') },
  { id: 'woodpeckers', key: 'Woodpeckers', label: i18n.t('groups.type.Woodpeckers') },
  { id: 'wagtails', key: 'Wagtails', label: i18n.t('groups.type.Wagtails') },
  { id: 'doves', key: 'Doves & Pigeons', label: i18n.t('groups.type.Doves & Pigeons') },
  { id: 'warblers', key: 'Warblers', label: i18n.t('groups.type.Warblers') },
  { id: 'waterside', key: 'Waterside Birds', label: i18n.t('groups.type.Waterside Birds') },
  { id: 'waders', key: 'Waders', label: i18n.t('groups.type.Waders') },
  { id: 'buntings', key: 'Buntings', label: i18n.t('groups.type.Buntings') },
  { id: 'aerial', key: 'Swifts, Swallows & Martins', label: i18n.t('groups.type.Swifts, Swallows & Martins') },
  { id: 'climbers', key: 'Woodland Climbers', label: i18n.t('groups.type.Woodland Climbers') },
  { id: 'gamebirds', key: 'Gamebirds', label: i18n.t('groups.type.Gamebirds') },
  { id: 'songbirds', key: 'Garden & Countryside Songbirds', label: i18n.t('groups.type.Garden & Countryside Songbirds') },
];

const TYPE_GROUP_BY_SPECIES = {
  Coal_Tit: 'tits',
  Eurasian_Blue_Tit: 'tits',
  Great_Tit: 'tits',
  Long_tailed_Tit: 'tits',

  Carrion_Crow: 'corvids',
  Eurasian_Jackdaw: 'corvids',
  Eurasian_Jay: 'corvids',
  Eurasian_Magpie: 'corvids',
  Common_Raven: 'corvids',
  Rook: 'corvids',

  Common_Buzzard: 'raptors',
  Common_Kestrel: 'raptors',
  Eurasian_Sparrowhawk: 'raptors',
  Red_Kite: 'raptors',

  Barn_Owl: 'owls',
  Tawny_Owl: 'owls',

  Canada_Goose: 'waterfowl',
  Mallard: 'waterfowl',
  Mute_Swan: 'waterfowl',

  Black_headed_Gull: 'gulls',
  European_Herring_Gull: 'gulls',

  Eurasian_Blackbird: 'thrushes',
  Mistle_Thrush: 'thrushes',
  Song_Thrush: 'thrushes',

  Eurasian_Bullfinch: 'finches',
  Eurasian_Chaffinch: 'finches',
  Eurasian_Greenfinch: 'finches',
  European_Goldfinch: 'finches',
  Common_Linnet: 'finches',

  Great_Spotted_Woodpecker: 'woodpeckers',
  Green_Woodpecker: 'woodpeckers',

  Grey_Wagtail: 'wagtails',
  Pied_Wagtail: 'wagtails',

  Eurasian_Collared_Dove: 'doves',
  Stock_Dove: 'doves',
  Woodpigeon: 'doves',

  Common_Blackcap: 'warblers',
  Common_Chiffchaff: 'warblers',
  Willow_Warbler: 'warblers',

  Common_Moorhen: 'waterside',
  Eurasian_Coot: 'waterside',
  Grey_Heron: 'waterside',
  Common_Kingfisher: 'waterside',

  Eurasian_Oystercatcher: 'waders',
  Northern_Lapwing: 'waders',

  Reed_Bunting: 'buntings',
  Yellowhammer: 'buntings',

  Barn_Swallow: 'aerial',
  Common_Swift: 'aerial',
  House_Martin: 'aerial',

  Eurasian_Nuthatch: 'climbers',
  Eurasian_Treecreeper: 'climbers',

  Common_Pheasant: 'gamebirds',
  Grey_Partridge: 'gamebirds',

  Dunnock: 'songbirds',
  Eurasian_Skylark: 'songbirds',
  Eurasian_Wren: 'songbirds',
  European_Robin: 'songbirds',
  European_Starling: 'songbirds',
  House_Sparrow: 'songbirds',
};

// Habitat trophy groupings — where you'd typically go looking for each
// species, not what family it belongs to (a Robin and a Blackbird are
// different families but the same "garden bird" trophy). Real birds cross
// habitats constantly; this pins each species to whichever habitat it's
// most classically associated with in UK birdwatching, same simplification
// tradeoff as the family groupings above. Every species belongs to exactly
// one habitat.
export const HABITAT_GROUPS = [
  { id: 'garden', key: 'Garden Birds', label: i18n.t('groups.habitat.Garden Birds') },
  { id: 'woodland', key: 'Woodland Birds', label: i18n.t('groups.habitat.Woodland Birds') },
  { id: 'wetland', key: 'Wetland & Coastal Birds', label: i18n.t('groups.habitat.Wetland & Coastal Birds') },
  { id: 'farmland', key: 'Farmland Birds', label: i18n.t('groups.habitat.Farmland Birds') },
];

const HABITAT_BY_SPECIES = {
  Carrion_Crow: 'garden',
  Common_Swift: 'garden',
  Dunnock: 'garden',
  Eurasian_Blackbird: 'garden',
  Eurasian_Blue_Tit: 'garden',
  Eurasian_Chaffinch: 'garden',
  Eurasian_Collared_Dove: 'garden',
  Eurasian_Greenfinch: 'garden',
  Eurasian_Jackdaw: 'garden',
  Eurasian_Magpie: 'garden',
  Eurasian_Wren: 'garden',
  European_Goldfinch: 'garden',
  European_Robin: 'garden',
  European_Starling: 'garden',
  Great_Tit: 'garden',
  House_Martin: 'garden',
  House_Sparrow: 'garden',
  Pied_Wagtail: 'garden',
  Song_Thrush: 'garden',
  Woodpigeon: 'garden',

  Coal_Tit: 'woodland',
  Common_Blackcap: 'woodland',
  Common_Chiffchaff: 'woodland',
  Eurasian_Bullfinch: 'woodland',
  Eurasian_Jay: 'woodland',
  Eurasian_Nuthatch: 'woodland',
  Eurasian_Sparrowhawk: 'woodland',
  Eurasian_Treecreeper: 'woodland',
  Great_Spotted_Woodpecker: 'woodland',
  Green_Woodpecker: 'woodland',
  Long_tailed_Tit: 'woodland',
  Tawny_Owl: 'woodland',
  Willow_Warbler: 'woodland',

  Black_headed_Gull: 'wetland',
  Canada_Goose: 'wetland',
  Common_Kingfisher: 'wetland',
  Common_Moorhen: 'wetland',
  Eurasian_Coot: 'wetland',
  Eurasian_Oystercatcher: 'wetland',
  European_Herring_Gull: 'wetland',
  Grey_Heron: 'wetland',
  Grey_Wagtail: 'wetland',
  Mallard: 'wetland',
  Mute_Swan: 'wetland',
  Reed_Bunting: 'wetland',

  Barn_Owl: 'farmland',
  Barn_Swallow: 'farmland',
  Common_Buzzard: 'farmland',
  Common_Kestrel: 'farmland',
  Common_Linnet: 'farmland',
  Common_Pheasant: 'farmland',
  Common_Raven: 'farmland',
  Eurasian_Skylark: 'farmland',
  Grey_Partridge: 'farmland',
  Mistle_Thrush: 'farmland',
  Northern_Lapwing: 'farmland',
  Red_Kite: 'farmland',
  Rook: 'farmland',
  Stock_Dove: 'farmland',
  Yellowhammer: 'farmland',
};

// Migratory status. Only species that genuinely leave Britain entirely for
// an African/southern-European winter count as Summer Visitors here —
// species that get winter influxes of continental birds while a resident
// UK population stays put (e.g. gulls, wagtails) are still Residents,
// since the species itself is present in the UK year-round.
export const MIGRATION_GROUPS = [
  { id: 'summer', key: 'Summer Visitors', label: i18n.t('groups.migration.Summer Visitors') },
  { id: 'resident', key: 'Year-round Residents', label: i18n.t('groups.migration.Year-round Residents') },
];

const SUMMER_VISITOR_IDS = new Set([
  'Barn_Swallow',
  'Common_Swift',
  'House_Martin',
  'Common_Chiffchaff',
  'Willow_Warbler',
  'Common_Blackcap',
]);

// Also used directly on a raw predicted_class from the API (see
// ResultCard.js/SwipeMatcher.js), hence the numeric-prefix strip. Falls back
// to the mechanical underscore replacement for any id without a translation
// entry, so an untranslated/unrecognised id still renders something sane.
export function formatSpeciesName(id) {
  if (!id) return '';
  const cleanId = id.replace(/^\d+\./, '');
  return i18n.t(`species.${cleanId}`, { defaultValue: cleanId.replace(/_/g, ' ') });
}

export const SPECIES = SPECIES_IDS.map((id) => ({
  id,
  name: formatSpeciesName(id),
  taxonId: TAXON_IDS[id],
  typeGroup: TYPE_GROUP_BY_SPECIES[id],
  habitat: HABITAT_BY_SPECIES[id],
  migration: SUMMER_VISITOR_IDS.has(id) ? 'summer' : 'resident',
})).sort((a, b) => a.name.localeCompare(b.name));

export const SPECIES_COUNT = SPECIES.length;
