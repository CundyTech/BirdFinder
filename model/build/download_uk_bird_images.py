#!/usr/bin/env python
"""Downloads real UK bird training photos from GBIF (api.gbif.org).

For each species below, looks up its GBIF taxon key by scientific name, then
paginates through UK occurrence records that have photos and downloads up to
--per-species images into <out>/<folder_name>/, alongside a manifest.csv
recording the source URL, license, and attribution for every image (GBIF
aggregates iNaturalist and other providers, mostly under CC licenses that
require attribution).

No API key needed. Uses only the standard library (urllib) so it runs with
any Python — no need for the TensorFlow environment.

Usage:
    python download_uk_bird_images.py
    python download_uk_bird_images.py --per-species 200 --species "Erithacus rubecula"
"""
import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"
USER_AGENT = "BirdFinderUK-DatasetBuilder/1.0 (personal hobby project)"
PAGE_SIZE = 100
REQUEST_DELAY_SECONDS = 0.3  # be polite to the public API

# (scientific name, folder-friendly common name). Common UK garden/countryside
# birds — a practical starting scope rather than the full ~600-species British
# list, most of which are rare vagrants with too few photos to train on.
SPECIES = [
    ("Erithacus rubecula", "European_Robin"),
    ("Cyanistes caeruleus", "Eurasian_Blue_Tit"),
    ("Parus major", "Great_Tit"),
    ("Turdus merula", "Eurasian_Blackbird"),
    ("Passer domesticus", "House_Sparrow"),
    ("Sturnus vulgaris", "European_Starling"),
    ("Pica pica", "Eurasian_Magpie"),
    ("Columba palumbus", "Woodpigeon"),
    ("Troglodytes troglodytes", "Eurasian_Wren"),
    ("Prunella modularis", "Dunnock"),
    ("Fringilla coelebs", "Eurasian_Chaffinch"),
    ("Carduelis carduelis", "European_Goldfinch"),
    ("Chloris chloris", "Eurasian_Greenfinch"),
    ("Pyrrhula pyrrhula", "Eurasian_Bullfinch"),
    ("Aegithalos caudatus", "Long_tailed_Tit"),
    ("Periparus ater", "Coal_Tit"),
    ("Sitta europaea", "Eurasian_Nuthatch"),
    ("Certhia familiaris", "Eurasian_Treecreeper"),
    ("Turdus philomelos", "Song_Thrush"),
    ("Turdus viscivorus", "Mistle_Thrush"),
    ("Coloeus monedula", "Eurasian_Jackdaw"),
    ("Corvus corone", "Carrion_Crow"),
    ("Corvus frugilegus", "Rook"),
    ("Corvus corax", "Common_Raven"),
    ("Garrulus glandarius", "Eurasian_Jay"),
    ("Sylvia atricapilla", "Common_Blackcap"),
    ("Phylloscopus collybita", "Common_Chiffchaff"),
    ("Phylloscopus trochilus", "Willow_Warbler"),
    ("Hirundo rustica", "Barn_Swallow"),
    ("Delichon urbicum", "House_Martin"),
    ("Apus apus", "Common_Swift"),
    ("Motacilla alba", "Pied_Wagtail"),
    ("Motacilla cinerea", "Grey_Wagtail"),
    ("Streptopelia decaocto", "Eurasian_Collared_Dove"),
    ("Columba oenas", "Stock_Dove"),
    ("Buteo buteo", "Common_Buzzard"),
    ("Accipiter nisus", "Eurasian_Sparrowhawk"),
    ("Falco tinnunculus", "Common_Kestrel"),
    ("Milvus milvus", "Red_Kite"),
    ("Strix aluco", "Tawny_Owl"),
    ("Tyto alba", "Barn_Owl"),
    ("Anas platyrhynchos", "Mallard"),
    ("Cygnus olor", "Mute_Swan"),
    ("Branta canadensis", "Canada_Goose"),
    ("Ardea cinerea", "Grey_Heron"),
    ("Gallinula chloropus", "Common_Moorhen"),
    ("Fulica atra", "Eurasian_Coot"),
    ("Alcedo atthis", "Common_Kingfisher"),
    ("Chroicocephalus ridibundus", "Black_headed_Gull"),
    ("Larus argentatus", "European_Herring_Gull"),
    ("Vanellus vanellus", "Northern_Lapwing"),
    ("Haematopus ostralegus", "Eurasian_Oystercatcher"),
    ("Dendrocopos major", "Great_Spotted_Woodpecker"),
    ("Picus viridis", "Green_Woodpecker"),
    ("Phasianus colchicus", "Common_Pheasant"),
    ("Perdix perdix", "Grey_Partridge"),
    ("Alauda arvensis", "Eurasian_Skylark"),
    ("Linaria cannabina", "Common_Linnet"),
    ("Emberiza citrinella", "Yellowhammer"),
    ("Emberiza schoeniclus", "Reed_Bunting"),
]


def http_get_json(url, params):
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_image(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"not an image ({content_type or 'no content-type'})")
        data = resp.read()
    dest_path.write_bytes(data)


def get_taxon_key(scientific_name):
    result = http_get_json(GBIF_MATCH_URL, {"name": scientific_name, "strict": "true"})
    if result.get("matchType") in (None, "NONE"):
        return None
    return result.get("usageKey")


def iter_occurrence_media(taxon_key, needed, seen_urls):
    """Yield (image_url, license, attribution) for a taxon, paginating as needed."""
    offset = 0
    collected = 0
    while collected < needed:
        page = http_get_json(GBIF_OCCURRENCE_URL, {
            "taxonKey": taxon_key,
            "country": "GB",
            "mediaType": "StillImage",
            "limit": PAGE_SIZE,
            "offset": offset,
        })
        results = page.get("results", [])
        if not results:
            return
        for record in results:
            license_ = record.get("license", "")
            attribution = record.get("recordedBy", "") or record.get("rightsHolder", "")
            for media in record.get("media", []):
                image_url = media.get("identifier")
                if not image_url or image_url in seen_urls:
                    continue
                seen_urls.add(image_url)
                yield image_url, license_, attribution
                collected += 1
                if collected >= needed:
                    return
        if page.get("endOfRecords"):
            return
        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY_SECONDS)


def run(species_list, per_species, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    manifest_file = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(manifest_file)
    if write_header:
        writer.writerow(["species", "filename", "source_url", "license", "attribution"])

    totals = {"species_done": 0, "images_downloaded": 0}

    for scientific_name, folder_name in species_list:
        species_dir = out_dir / folder_name
        species_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(species_dir.glob(f"{folder_name}_*.jpg"))
        if len(existing) >= per_species:
            print(f"[skip] {folder_name}: already have {len(existing)} images")
            totals["species_done"] += 1
            continue

        print(f"[fetch] {folder_name} ({scientific_name})...")
        try:
            taxon_key = get_taxon_key(scientific_name)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  !! GBIF match request failed: {e}")
            continue
        if taxon_key is None:
            print(f"  !! No GBIF match for '{scientific_name}', skipping")
            continue

        needed = per_species - len(existing)
        seen_urls = set()
        downloaded = 0
        try:
            for image_url, license_, attribution in iter_occurrence_media(taxon_key, needed, seen_urls):
                index = len(existing) + downloaded
                dest = species_dir / f"{folder_name}_{index:04d}.jpg"
                try:
                    download_image(image_url, dest)
                    writer.writerow([folder_name, dest.name, image_url, license_, attribution])
                    downloaded += 1
                except Exception as e:
                    print(f"  !! Failed to download {image_url}: {e}")
                time.sleep(REQUEST_DELAY_SECONDS)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  !! GBIF occurrence search failed: {e}")

        manifest_file.flush()
        print(f"  -> downloaded {downloaded} images ({len(existing) + downloaded} total)")
        totals["species_done"] += 1
        totals["images_downloaded"] += downloaded

    manifest_file.close()
    print(f"\nDone. {totals['species_done']}/{len(species_list)} species processed, "
          f"{totals['images_downloaded']} new images. Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-species", type=int, default=150, help="target images per species")
    parser.add_argument("--out", default=None, help="output directory (default: ../images/uk_birds)")
    parser.add_argument("--species", default=None, help="scientific name of a single species to (re)fetch")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out) if args.out else (base_dir / ".." / "images" / "uk_birds").resolve()

    species_list = SPECIES
    if args.species:
        species_list = [s for s in SPECIES if s[0].lower() == args.species.lower()]
        if not species_list:
            print(f"'{args.species}' not found in SPECIES list")
            return

    run(species_list, args.per_species, out_dir)


if __name__ == "__main__":
    main()
