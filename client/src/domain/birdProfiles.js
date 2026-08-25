// Curated "fun profile" content for the life-list facts card: general
// ornithological knowledge for common, well-documented UK species, not
// pulled from a live API like the rarity/taxonomy data elsewhere in this
// app (see species.js). Worth a sanity check if anything here reads oddly
// for a specific species.
export const BIRD_PROFILES = {
  Barn_Owl: {
    fact: 'Barn Owls can hear a mouse rustling in grass from over 30 metres away thanks to asymmetrical ear openings.',
    prey: 'Small mammals, mainly field voles, mice and shrews.',
    predators: "Adults have few predators, but eggs and chicks may be taken by other owls or foxes.",
  },
  Barn_Swallow: {
    fact: 'A Barn Swallow migrates around 10,000km each way between the UK and southern Africa every year.',
    prey: 'Flying insects, caught entirely on the wing.',
    predators: 'Sparrowhawks and Hobbies; cats and magpies can take nestlings.',
  },
  Black_headed_Gull: {
    fact: "Despite the name, a Black-headed Gull's 'black' head is actually dark chocolate-brown, and it's only there in summer.",
    prey: 'Insects, worms, fish and scraps. A very opportunistic feeder.',
    predators: 'Foxes and larger gulls raid nests and take chicks.',
  },
  Canada_Goose: {
    fact: 'Introduced to Britain in the 17th century as an ornamental bird, Canada Geese are now one of our most common waterfowl.',
    prey: 'Grasses, grain and aquatic plants.',
    predators: 'Foxes take goslings; adults have few natural predators.',
  },
  Carrion_Crow: {
    fact: "Carrion Crows are remarkably intelligent. They've been observed using tools and recognising individual human faces.",
    prey: 'Omnivorous: carrion, insects, eggs, small animals and food scraps.',
    predators: 'Eggs and chicks taken by other corvids and foxes; adults occasionally by birds of prey.',
  },
  Coal_Tit: {
    fact: 'A Coal Tit hides hundreds of seeds in autumn to retrieve through winter, and remembers where most of them are.',
    prey: 'Insects and spiders in summer, seeds and nuts in winter.',
    predators: 'Sparrowhawks; nests raided by weasels and woodpeckers.',
  },
  Common_Blackcap: {
    fact: "Some Blackcaps that breed in Germany now spend winter in British gardens instead of flying to Africa, a migration route that's evolved within living memory.",
    prey: 'Insects in summer, switching to berries and fruit in autumn and winter.',
    predators: 'Sparrowhawks; nests raided by jays and squirrels.',
  },
  Common_Buzzard: {
    fact: "The Common Buzzard's mewing call is often mistaken for a cat, making it one of the most frequently misidentified bird sounds in Britain.",
    prey: 'Small mammals, especially rabbits, plus carrion and earthworms.',
    predators: 'Adults have virtually no natural predators in the UK.',
  },
  Common_Chiffchaff: {
    fact: "The Chiffchaff is named for its song, a simple, repetitive 'chiff-chaff' that's one of the first signs of spring.",
    prey: 'Small insects, especially midges and aphids, gleaned from leaves.',
    predators: 'Sparrowhawks; nests vulnerable to stoats and weasels.',
  },
  Common_Kestrel: {
    fact: 'Kestrels can see ultraviolet light, which helps them spot the urine trails voles leave behind, like a glowing trail to their next meal.',
    prey: 'Small mammals, especially voles, plus insects and small birds.',
    predators: 'Larger raptors such as Goshawks; eggs taken by crows.',
  },
  Common_Kingfisher: {
    fact: 'A Kingfisher can dive at up to 25mph and closes a third eyelid to protect its eyes on impact with the water.',
    prey: 'Small fish, especially minnows and sticklebacks.',
    predators: 'Sparrowhawks; mink are a serious threat to riverside nests.',
  },
  Common_Linnet: {
    fact: "The male Linnet's rosy-pink breast comes from its diet. Caged birds kept by Victorian collectors famously faded to dull brown without the right seeds.",
    prey: 'Seeds, especially from weeds like chickweed and dandelion.',
    predators: 'Sparrowhawks and Merlins.',
  },
  Common_Moorhen: {
    fact: 'Moorhen chicks from an earlier brood often help their parents feed and raise the next batch of chicks.',
    prey: 'Aquatic plants, seeds, insects and snails.',
    predators: 'Foxes, mink and pike take chicks; herons occasionally take adults.',
  },
  Common_Pheasant: {
    fact: "Common Pheasants aren't native to Britain. They were likely introduced by the Romans, and millions are still reared and released for shooting today.",
    prey: 'Seeds, grain, berries and insects.',
    predators: 'Foxes are the main predator, especially of eggs and chicks.',
  },
  Common_Raven: {
    fact: "Ravens are among the smartest birds alive, capable of planning ahead and solving multi-step puzzles rivalling a young child's abilities.",
    prey: 'Omnivorous scavenger: carrion, small animals, eggs and food waste.',
    predators: 'Adults have almost no natural predators; eagles occasionally.',
  },
  Common_Swift: {
    fact: 'A Swift can stay airborne for up to 10 months without landing once, even sleeping on the wing.',
    prey: 'Flying insects and airborne spiders, caught entirely in flight.',
    predators: 'Hobbies are one of the few birds fast enough to catch a Swift.',
  },
  Dunnock: {
    fact: 'Dunnocks have one of the most complicated mating systems of any British bird, regularly involving multiple mates for both males and females.',
    prey: 'Small insects and spiders, plus seeds in winter.',
    predators: 'Sparrowhawks; nests raided by cats and weasels.',
  },
  Eurasian_Blackbird: {
    fact: "A male Blackbird's song is so complex that individual birds can be recognised by their own personal 'signature tune'.",
    prey: 'Worms, insects and berries, foraged by flicking leaf litter aside.',
    predators: 'Sparrowhawks and domestic cats.',
  },
  Eurasian_Blue_Tit: {
    fact: 'Blue Tits can see ultraviolet light, and their blue crown feathers glow under UV, likely a signal of fitness to potential mates.',
    prey: 'Caterpillars and insects in summer, seeds and nuts in winter.',
    predators: 'Sparrowhawks; nests raided by Great Spotted Woodpeckers and weasels.',
  },
  Eurasian_Bullfinch: {
    fact: 'Bullfinches were once persecuted as orchard pests for stripping fruit tree buds, getting through around 30 a minute.',
    prey: 'Buds, seeds and berries; insects fed to chicks.',
    predators: 'Sparrowhawks.',
  },
  Eurasian_Chaffinch: {
    fact: 'Chaffinches have regional song "dialects": birds from different areas of Britain sing slightly different versions of the same tune.',
    prey: 'Seeds and insects, foraging mostly on the ground.',
    predators: 'Sparrowhawks.',
  },
  Eurasian_Collared_Dove: {
    fact: "Collared Doves didn't breed in Britain until the 1950s. Within 50 years they'd spread across the entire country from a single colonisation event.",
    prey: 'Seeds and grain, often around bird feeders and farmyards.',
    predators: 'Sparrowhawks; magpies take eggs and chicks.',
  },
  Eurasian_Coot: {
    fact: 'Coots are famously aggressive over territory, and fights between rival coots can occasionally end in one bird drowning the other.',
    prey: 'Aquatic plants, insects and small fish.',
    predators: 'Foxes, mink and pike; eggs taken by crows.',
  },
  Eurasian_Greenfinch: {
    fact: "A wheezy call and flash of yellow in the wings gave the Greenfinch its old country name, 'green linnet'.",
    prey: 'Seeds, particularly favouring sunflower hearts at garden feeders.',
    predators: 'Sparrowhawks.',
  },
  Eurasian_Jackdaw: {
    fact: 'Jackdaws can recognise individual human faces and communicate danger to each other with specific alarm calls.',
    prey: 'Omnivorous: insects, seeds, scraps and eggs of other birds.',
    predators: 'Sparrowhawks and larger owls; foxes take eggs from low nests.',
  },
  Eurasian_Jay: {
    fact: 'A single Jay can bury thousands of acorns each autumn, and the ones it forgets help plant new oak woodlands.',
    prey: 'Acorns and seeds, plus insects and occasionally eggs or chicks.',
    predators: 'Goshawks; foxes take eggs and young.',
  },
  Eurasian_Magpie: {
    fact: 'Magpies are one of the few non-mammal species that can recognise themselves in a mirror, a sign of self-awareness.',
    prey: 'Omnivorous: insects, carrion, eggs and food scraps.',
    predators: 'Goshawks; few other natural predators.',
  },
  Eurasian_Nuthatch: {
    fact: 'Nuthatches are the only British bird that can walk head-first down a tree trunk.',
    prey: 'Insects and spiders in summer, nuts and seeds wedged into bark in winter.',
    predators: 'Sparrowhawks; nests raided by Great Spotted Woodpeckers.',
  },
  Eurasian_Oystercatcher: {
    fact: "An Oystercatcher's chisel-shaped bill is a specialist tool for prising open mussels and cockles on the shore.",
    prey: 'Shellfish such as mussels and cockles, plus worms.',
    predators: 'Foxes and gulls take eggs and chicks.',
  },
  Eurasian_Skylark: {
    fact: 'A Skylark can sing continuously for several minutes while hovering high overhead, rarely repeating its song exactly the same way twice.',
    prey: 'Seeds and insects, foraging on the ground in open fields.',
    predators: 'Kestrels and Merlins; eggs vulnerable to ground predators like foxes.',
  },
  Eurasian_Sparrowhawk: {
    fact: 'A female Sparrowhawk is almost twice the size of a male, one of the biggest size differences between sexes of any British bird.',
    prey: 'Small to medium birds, caught in fast, low ambush flights through hedges and gardens.',
    predators: 'Goshawks occasionally; otherwise very few natural predators.',
  },
  Eurasian_Treecreeper: {
    fact: 'A Treecreeper always spirals upward when climbing a tree, then flies to the base of the next one to start again.',
    prey: 'Insects and spiders picked from bark crevices with its curved bill.',
    predators: 'Sparrowhawks.',
  },
  Eurasian_Wren: {
    fact: "Despite weighing about as much as a £1 coin, the Wren's song is one of the loudest of any British bird relative to its size.",
    prey: 'Insects and spiders, foraged from low cover and undergrowth.',
    predators: 'Sparrowhawks and cats, though cold winters kill far more Wrens than predators do.',
  },
  European_Goldfinch: {
    fact: "A group of Goldfinches is called a 'charm', a fitting name for one of Britain's most colourful garden birds.",
    prey: 'Seeds, especially from thistles and teasels, using its fine pointed bill.',
    predators: 'Sparrowhawks.',
  },
  European_Herring_Gull: {
    fact: "Herring Gulls perform a 'rain dance', pattering their feet on grass to mimic rainfall and trick earthworms to the surface.",
    prey: 'Fish, shellfish, carrion and food scraps. A highly opportunistic feeder.',
    predators: 'Adults have few predators; eggs and chicks taken by foxes and other gulls.',
  },
  European_Robin: {
    fact: 'Robins are fiercely territorial and will attack almost anything red intruding on their patch, even a tuft of red feathers on a stick.',
    prey: 'Insects, worms and spiders, plus berries and seeds in winter.',
    predators: 'Domestic cats and Sparrowhawks.',
  },
  European_Starling: {
    fact: "Huge winter flocks of Starlings form swirling 'murmurations' of thousands of birds, moving as one to confuse predators.",
    prey: 'Insects, especially leatherjackets found by probing lawns, plus fruit and scraps.',
    predators: 'Sparrowhawks, which murmurations are partly a defence against.',
  },
  Great_Spotted_Woodpecker: {
    fact: "A woodpecker's skull has spongy, shock-absorbing bone that protects its brain through thousands of pecks a day.",
    prey: 'Insects and larvae found under bark, plus nuts and other birds’ eggs and chicks.',
    predators: 'Sparrowhawks.',
  },
  Great_Tit: {
    fact: 'Great Tits have dozens of different calls, and populations in noisy cities have been found to sing at a higher pitch to be heard over traffic.',
    prey: 'Caterpillars and insects in summer, seeds and nuts in winter.',
    predators: 'Sparrowhawks; nests raided by weasels and woodpeckers.',
  },
  Green_Woodpecker: {
    fact: "The Green Woodpecker's loud, laughing call earned it the old country name 'yaffle'.",
    prey: 'Ants, which it digs out of lawns and mounds with a tongue up to 10cm long.',
    predators: 'Sparrowhawks and Goshawks.',
  },
  Grey_Heron: {
    fact: 'A Heron can stand motionless for so long waiting to strike that it’s easy to mistake it for a garden ornament.',
    prey: 'Fish, amphibians and small mammals, speared with a lightning-fast strike.',
    predators: 'Adults have very few predators; foxes may take chicks at ground-level colonies.',
  },
  Grey_Partridge: {
    fact: 'Grey Partridge numbers have fallen by over 90% in Britain since the 1970s due to changes in farming, making it a real conservation priority.',
    prey: "Seeds and insects, especially important for feeding chicks in their first weeks.",
    predators: 'Foxes, and birds of prey take eggs, chicks and adults.',
  },
  Grey_Wagtail: {
    fact: 'Despite the name, the Grey Wagtail has a bright yellow belly and is often mistaken for the scarcer Yellow Wagtail.',
    prey: 'Insects, especially flies, caught along fast-flowing streams and rivers.',
    predators: 'Sparrowhawks.',
  },
  House_Martin: {
    fact: 'House Martins build their mud nests entirely from small beakfuls of wet mud, gathered one mouthful at a time from puddles.',
    prey: 'Flying insects, caught on the wing high above ground.',
    predators: 'Sparrowhawks and Hobbies.',
  },
  House_Sparrow: {
    fact: 'House Sparrow numbers in UK towns and cities have fallen by around 70% since the 1970s, for reasons still not fully understood.',
    prey: 'Seeds and grain, plus insects fed to chicks.',
    predators: 'Domestic cats and Sparrowhawks.',
  },
  Long_tailed_Tit: {
    fact: "A Long-tailed Tit's nest is an engineering marvel: a stretchy ball woven from moss, spider silk and over 1,500 feathers for insulation.",
    prey: 'Small insects and spiders, gleaned from twigs in constantly moving flocks.',
    predators: 'Nest predation by crows and weasels is a major cause of failure; adults taken by Sparrowhawks.',
  },
  Mallard: {
    fact: "A female Mallard's quack famously produces an echo, while the male's call is a quieter, raspier sound.",
    prey: 'Aquatic plants, seeds, insects and small invertebrates.',
    predators: 'Foxes take eggs and ducklings; mink are a serious nest predator.',
  },
  Mistle_Thrush: {
    fact: 'The Mistle Thrush earned its name from its fondness for mistletoe berries, which it fiercely guards from other birds all winter.',
    prey: 'Worms, insects and berries, especially holly and mistletoe.',
    predators: 'Sparrowhawks.',
  },
  Mute_Swan: {
    fact: "Despite the name, Mute Swans aren't silent. They hiss, grunt, and their wings make a distinctive throbbing sound in flight.",
    prey: 'Aquatic plants, mainly, pulled up from underwater with their long necks.',
    predators: 'Foxes take cygnets; adults have very few predators.',
  },
  Northern_Lapwing: {
    fact: 'A Lapwing will fake a broken wing to lure predators away from its ground nest, then fly off once the danger has followed far enough.',
    prey: 'Insects and worms, picked from open farmland and grassland.',
    predators: 'Foxes and crows are major nest predators.',
  },
  Pied_Wagtail: {
    fact: 'Pied Wagtails often gather to roost together in huge numbers in city centres, drawn to the extra warmth of buildings and streetlights.',
    prey: 'Insects, caught on the ground or in short aerial dashes.',
    predators: 'Sparrowhawks and domestic cats.',
  },
  Red_Kite: {
    fact: 'Red Kites were nearly extinct in Britain by the 1930s, surviving in a handful of Welsh valleys. Reintroduction since 1989 has made them a common sight again.',
    prey: 'Carrion and small animals; famous for scavenging roadkill.',
    predators: 'Adults have very few natural predators in the UK.',
  },
  Reed_Bunting: {
    fact: 'Male Reed Buntings look strikingly different in summer and winter: a smart black-and-white head in the breeding season fades to streaky brown for winter.',
    prey: 'Seeds in winter, insects in the breeding season.',
    predators: 'Sparrowhawks and Marsh Harriers.',
  },
  Rook: {
    fact: 'Rooks nest in noisy colonies called rookeries, sometimes with hundreds of nests clustered in the same treetops.',
    prey: 'Mainly insect larvae and worms dug from soil, plus grain and seeds.',
    predators: 'Adults have few natural predators; eggs and chicks taken by other corvids.',
  },
  Song_Thrush: {
    fact: "A Song Thrush uses a favourite stone as an 'anvil' to smash open snail shells. You can often find a scatter of broken shells nearby.",
    prey: 'Snails, worms and insects, plus berries in autumn.',
    predators: 'Sparrowhawks and domestic cats.',
  },
  Stock_Dove: {
    fact: 'Stock Doves nest in holes in old trees, cliffs and even rabbit burrows, unlike their cousin the Woodpigeon, which builds an open nest.',
    prey: 'Seeds and grain, foraged from farmland and stubble fields.',
    predators: 'Sparrowhawks and Goshawks.',
  },
  Tawny_Owl: {
    fact: "The classic 'twit-twoo' owl call is actually two owls duetting: the female makes the sharp 'kewick' and the male the soft 'hoo-hoo-oo'.",
    prey: 'Small mammals, especially wood mice and voles, plus small birds.',
    predators: 'Goshawks; adults have few other natural predators.',
  },
  Willow_Warbler: {
    fact: "The UK's most numerous long-distance migrant, the Willow Warbler crosses the Sahara twice a year to winter in sub-Saharan Africa.",
    prey: 'Small insects, especially aphids, gleaned from leaves.',
    predators: 'Sparrowhawks; nests vulnerable to weasels and stoats.',
  },
  Woodpigeon: {
    fact: 'Woodpigeons are the heaviest and most numerous pigeon in Britain, and can raise young in almost any month of the year.',
    prey: 'Seeds, grain, leaves and crops. Considered a significant agricultural pest.',
    predators: 'Sparrowhawks and Peregrine Falcons.',
  },
  Yellowhammer: {
    fact: "The Yellowhammer's song is famously remembered by the phrase 'a little bit of bread and no cheese'.",
    prey: 'Seeds, especially in winter, and insects fed to chicks in summer.',
    predators: 'Sparrowhawks.',
  },
};

// A second, distinct fact per species — unlocked as a milestone reward
// rather than shown by default (see rewardsSlice.js's deepDiveLoreUnlocked).
// Deliberately a different angle from BIRD_PROFILES' `fact` field (name
// origin, conservation history, etc.) rather than a repeat of it.
export const DEEP_DIVE = {
  Barn_Owl: 'Barn Owl numbers crashed by around 70% across Britain during the 20th century as barns were converted and old trees felled, though nest box schemes have since helped many local populations recover.',
  Barn_Swallow: "The Barn Swallow's forked tail streamers are longer in males, and females tend to choose mates with the longest, most symmetrical tails.",
  Black_headed_Gull: "Black-headed Gulls are surprisingly inland birds for a 'seagull', breeding in colonies on lakes, gravel pits and marshes far from the coast.",
  Canada_Goose: 'Canada Geese mate for life and can live for over 20 years, with pairs often returning to the same nest site year after year.',
  Carrion_Crow: 'Carrion Crows will sometimes drop hard-shelled food like walnuts onto roads for passing cars to crack open.',
  Coal_Tit: "The smallest of Britain's tits, the Coal Tit has a distinctive white stripe down the back of its head and favours conifer woodland over broadleaf.",
  Common_Blackcap: 'Male Blackcaps have a black cap while females and young birds have a reddish-brown one, making the species easy to sex at a glance.',
  Common_Buzzard: "Once confined to the west of Britain by centuries of persecution, the Common Buzzard has recolonised almost the whole country since the 1990s and is now the UK's most widespread bird of prey.",
  Common_Chiffchaff: 'A small but growing number of Chiffchaffs now overwinter in the UK rather than migrating to Africa, something that was rare before recent decades of milder winters.',
  Common_Kestrel: 'Kestrels hunt by hovering with remarkably still heads, holding position in open air even in a stiff breeze while scanning the ground below.',
  Common_Kingfisher: 'A Kingfisher pair can raise several broods in a summer, tunnelling a new nest burrow up to a metre long into a riverbank each time.',
  Common_Linnet: 'Linnets were among the most popular caged songbirds in Victorian Britain, and wild-trapping for the pet trade once seriously depleted populations.',
  Common_Moorhen: "The bright red frontal shield above a Moorhen's bill is used in territorial disputes, and tends to grow larger and brighter in more dominant birds.",
  Common_Pheasant: 'Around 40 million pheasants are released into the British countryside each year for shooting, making them the most numerous bird species in the UK by some estimates.',
  Common_Raven: 'The largest member of the crow family in Britain, the Common Raven was once a common scavenger in cities but was driven to remote uplands by centuries of persecution.',
  Common_Swift: 'Swifts arrive in Britain later and leave earlier than almost any other summer migrant, often present for only around three months of the year.',
  Dunnock: "Despite its plain, sparrow-like looks, the Dunnock is not a sparrow at all — it belongs to the accentor family and was once known as the 'hedge sparrow'.",
  Eurasian_Blackbird: 'Male Blackbirds are all black with a bright yellow-orange bill, while females and young birds are dark brown, often confusing new birdwatchers.',
  Eurasian_Blue_Tit: 'A single Blue Tit brood can eat many thousands of caterpillars before fledging, making them valuable natural pest control in gardens and orchards.',
  Eurasian_Bullfinch: "The Bullfinch's stocky build and thick bill are specialised for crushing seeds and stripping buds, and pairs are often seen feeding together year-round.",
  Eurasian_Chaffinch: "One of Britain's most common garden birds, the Chaffinch shows a flash of white on its wings and tail that becomes obvious in flight.",
  Eurasian_Collared_Dove: 'Collared Doves are believed to have spread across Europe from Asia Minor over the last century, one of the fastest natural range expansions ever recorded in a bird.',
  Eurasian_Coot: 'Coot chicks hatch with a striking orange-red head and tufts of down, quite unlike the plain black-and-white plumage of the adults.',
  Eurasian_Greenfinch: 'Greenfinch numbers fell sharply in Britain after the 2000s due to trichomonosis, a parasitic disease easily spread at crowded garden feeding stations.',
  Eurasian_Jackdaw: 'Jackdaws are the smallest member of the crow family in Britain and nest in loose colonies in chimneys, cliffs and tree holes.',
  Eurasian_Jay: 'Despite its brightly coloured wing patch, the Jay is a notoriously shy woodland bird, more often heard as a harsh screech than actually seen.',
  Eurasian_Magpie: "A folk rhyme, 'one for sorrow, two for joy', has attached superstition to Magpie sightings in Britain for centuries.",
  Eurasian_Nuthatch: 'Nuthatches plaster mud around the entrance of their nest hole to shrink it, keeping out larger birds that might otherwise evict them.',
  Eurasian_Oystercatcher: 'Despite the name, Oystercatchers rarely eat oysters — mussels and cockles make up the bulk of their diet on British shores.',
  Eurasian_Skylark: 'Skylark numbers on British farmland have roughly halved since the 1970s, largely linked to changes in autumn-sown crops that leave less nesting cover.',
  Eurasian_Sparrowhawk: 'Sparrowhawks were hit hard by organochlorine pesticides in the mid-20th century and vanished from large parts of Britain before recovering after the chemicals were banned.',
  Eurasian_Treecreeper: 'A Treecreeper roosts by pressing itself into a shallow cavity in soft bark, sometimes even hollowing out a snug niche in the trunk of a giant redwood.',
  Eurasian_Wren: 'In some parts of Britain, dozens of wrens have been found huddled together in a single nest box on a cold night, sharing body heat to survive.',
  European_Goldfinch: 'Goldfinches were traditionally associated with the Passion of Christ in European art, often shown perched near the infant Jesus in old religious paintings.',
  European_Herring_Gull: 'Herring Gulls have adapted so well to towns and cities that many British populations now nest almost entirely on rooftops rather than sea cliffs.',
  European_Robin: "The Robin's association with Christmas cards dates back to Victorian postmen, who wore red waistcoats and were nicknamed 'robins' by the public.",
  European_Starling: "A Starling can mimic an extraordinary range of sounds, from other birds' songs to car alarms and mobile phone ringtones.",
  Great_Spotted_Woodpecker: 'Great Spotted Woodpeckers drum on tree trunks up to 40 times a second to declare territory, rather than to find food.',
  Great_Tit: "The Great Tit is the largest of Britain's tit species and its two-note 'teacher-teacher' song is one of the earliest signs of spring.",
  Green_Woodpecker: 'Unlike most woodpeckers, the Green Woodpecker spends much of its time feeding on the ground rather than in trees.',
  Grey_Heron: "Grey Herons nest colonially in treetop 'heronries', some of which have been used continuously by generations of herons for well over a century.",
  Grey_Partridge: 'Once one of the most common farmland birds in Britain, the Grey Partridge is now a conservation priority after steep declines linked to modern arable farming.',
  Grey_Wagtail: 'Grey Wagtails favour fast-flowing streams and weirs, and can often be seen bobbing their long tail up and down on riverside rocks.',
  House_Martin: 'House Martins have declined so sharply in parts of Britain that the species is now on the conservation Red List, despite still nesting on houses across the country.',
  House_Sparrow: 'House Sparrows live in loose colonies and rarely stray far from the buildings where they hatched, sometimes spending their entire lives within a few hundred metres.',
  Long_tailed_Tit: 'Long-tailed Tits are highly social outside the breeding season, moving through hedgerows in tight family flocks and roosting huddled together for warmth.',
  Mallard: 'The Mallard is the ancestor of almost all domestic duck breeds, and wild birds readily interbreed with farmyard ducks, producing a huge range of plumage variations.',
  Mistle_Thrush: "The Mistle Thrush is sometimes called the 'stormcock' for its habit of singing loudly from treetops even in wild, wet weather.",
  Mute_Swan: 'Mute Swans in Britain are traditionally considered property of the Crown, and an annual ceremony called Swan Upping still counts and marks swans on the Thames.',
  Northern_Lapwing: "The Lapwing's rounded wings produce a distinctive throbbing sound in flight, which combined with its wheeling display gave rise to the old country name 'peewit'.",
  Pied_Wagtail: "Pied Wagtails are the UK's resident race of the White Wagtail, distinguished from their continental cousins by a darker back.",
  Red_Kite: "The Red Kite's deeply forked tail and effortless soaring flight make it one of the easiest British raptors to identify at a distance.",
  Reed_Bunting: 'Reed Buntings have expanded beyond their traditional wetland habitat in recent decades and are now regularly seen on farmland and even in gardens in winter.',
  Rook: "A patch of pale, featherless skin around the base of a Rook's bill distinguishes it from the very similar Carrion Crow at close range.",
  Song_Thrush: 'Song Thrush numbers fell by around half in Britain during the late 20th century, with intensified farming and tidier gardens both blamed for reducing the snails and worms they rely on.',
  Stock_Dove: "The Stock Dove lacks the white neck patch and wing markings of the Woodpigeon, and its call is a soft, repetitive double-note rather than a cooing coo-coo-coo.",
  Tawny_Owl: "Britain's most common owl, the Tawny Owl rarely leaves woodland, making its call far more familiar to most people than its appearance.",
  Willow_Warbler: 'Willow Warblers can be told apart from the near-identical Chiffchaff mainly by song, since the two species look almost indistinguishable in the field.',
  Woodpigeon: "A Woodpigeon feeds its squabs on 'crop milk', a nutrient-rich secretion produced in its crop, much like pigeons and doves worldwide.",
  Yellowhammer: 'Yellowhammers favour hedgerows and field margins, and their steep decline on intensive farmland has made them a key indicator species for farmland bird conservation.',
};

// Only species that genuinely leave Britain for winter (see species.js's
// MIGRATION_GROUPS) have a route. Everyone else is a year-round resident,
// shown as plain text instead of an empty/misleading map.
export const MIGRATION_ROUTES = {
  Barn_Swallow: { breeding: 'UK (Apr–Sep)', wintering: 'Southern Africa' },
  Common_Swift: { breeding: 'UK (May–Aug)', wintering: 'Central Africa' },
  House_Martin: { breeding: 'UK (Apr–Sep)', wintering: 'Sub-Saharan Africa' },
  Common_Chiffchaff: { breeding: 'UK (Mar–Oct)', wintering: 'Iberia & North Africa' },
  Willow_Warbler: { breeding: 'UK (Apr–Aug)', wintering: 'Sub-Saharan Africa' },
  Common_Blackcap: { breeding: 'UK (Apr–Sep)', wintering: 'Iberia & North Africa' },
};
