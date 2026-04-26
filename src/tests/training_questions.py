# training_questions.py
# 40 additional myth-busting questions in the same format as the original set

ADDITIONAL_QUESTIONS = [
    # ==================== GEOGRAPHY (6) ====================
    {
        "question": "Which country has the longest coastline in the world?",
        "correct_answer": "Canada has the world's longest coastline, measuring over 202,080 kilometers due to its numerous islands and fjords. Indonesia and Norway also have very long coastlines.",
        "wrong_answer": "Russia, because it's the largest country by land area",
        "category": "geography",
        "subcategory": "physical_features"
    },
    {
        "question": "Is the Amazon River the longest river in the world?",
        "correct_answer": "The Nile is generally considered the longest river at about 6,650 km, though the Amazon is very close and some measurements dispute the title. The Amazon is undisputedly the largest by water volume.",
        "wrong_answer": "Yes, the Amazon is definitely the longest",
        "category": "geography",
        "subcategory": "physical_features"
    },
    {
        "question": "What is the capital of South Africa?",
        "correct_answer": "South Africa has three capital cities: Pretoria (executive), Cape Town (legislative), and Bloemfontein (judicial). Johannesburg is the largest city but not a capital.",
        "wrong_answer": "Johannesburg",
        "category": "geography",
        "subcategory": "capitals"
    },
    {
        "question": "Is Greenland a continent?",
        "correct_answer": "No, Greenland is the world's largest island. It is geologically part of North America but politically an autonomous territory of Denmark. Continents are defined by tectonic plates and size.",
        "wrong_answer": "Yes, Greenland is its own continent",
        "category": "geography",
        "subcategory": "physical_features"
    },
    {
        "question": "Are the Great Lakes the largest freshwater lakes by volume?",
        "correct_answer": "No. Lake Baikal in Siberia holds about 20% of the world's unfrozen surface freshwater—more than all the Great Lakes combined. Lake Superior is the largest by surface area.",
        "wrong_answer": "Yes, the Great Lakes contain the most freshwater",
        "category": "geography",
        "subcategory": "physical_features"
    },
    {
        "question": "How many continents are there?",
        "correct_answer": "The number of continents varies by educational tradition: seven (North America, South America, Europe, Asia, Africa, Australia/Oceania, Antarctica) is common, but some regions teach six (combining the Americas) or five (combining Americas and omitting Antarctica).",
        "wrong_answer": "Exactly seven, and that's universal",
        "category": "geography",
        "subcategory": "political"
    },

    # ==================== HISTORY (7) ====================
    {
        "question": "Did Christopher Columbus discover America?",
        "correct_answer": "No. Indigenous peoples had lived in the Americas for millennia. Norse explorer Leif Erikson reached North America around 1000 CE. Columbus's voyages initiated sustained European contact but did not 'discover' the continents.",
        "wrong_answer": "Yes, Columbus discovered America in 1492",
        "category": "history",
        "subcategory": "exploration"
    },
    {
        "question": "Did Nero fiddle while Rome burned?",
        "correct_answer": "No. The fiddle did not exist in ancient Rome. Nero was not even in Rome when the fire started. He returned and organized relief efforts. The story is likely political slander.",
        "wrong_answer": "Yes, he played his fiddle and watched",
        "category": "history",
        "subcategory": "myth"
    },
    {
        "question": "Did medieval Europeans believe the Earth was flat?",
        "correct_answer": "No. Educated Europeans knew the Earth was spherical. The myth was popularized in the 19th century. Ancient Greeks had calculated Earth's circumference, and this knowledge was never lost in the Middle Ages.",
        "wrong_answer": "Yes, everyone thought it was flat until Columbus",
        "category": "history",
        "subcategory": "myth"
    },
    {
        "question": "Did Paul Revere shout 'The British are coming!' on his midnight ride?",
        "correct_answer": "No. Revere's mission was stealthy; he warned that 'the Regulars are coming out.' Many colonists still considered themselves British, so shouting 'The British are coming' would have been confusing.",
        "wrong_answer": "Yes, that's exactly what he yelled",
        "category": "history",
        "subcategory": "misattributed_quote"
    },
    {
        "question": "Did the 1938 'War of the Worlds' radio broadcast cause widespread mass panic?",
        "correct_answer": "Contemporary research shows that claims of mass hysteria were greatly exaggerated by newspapers seeking to discredit radio as a medium. Very few people actually panicked.",
        "wrong_answer": "Yes, millions of Americans fled their homes in terror",
        "category": "history",
        "subcategory": "myth"
    },
    {
        "question": "Did Henry Ford invent the assembly line?",
        "correct_answer": "No. The assembly line was used earlier in other industries, such as meatpacking and firearms manufacturing. Ford's innovation was the moving assembly line with conveyor belts, which dramatically increased efficiency.",
        "wrong_answer": "Yes, he invented it from scratch",
        "category": "history",
        "subcategory": "invention"
    },
    {
        "question": "Was the Wild West full of daily gunfights and shootouts?",
        "correct_answer": "No. Hollywood exaggerates. Actual gunfights were rare, and towns often had stricter gun control laws than modern cities. Murders and violent crime were relatively low.",
        "wrong_answer": "Yes, it was constant dueling and shootouts",
        "category": "history",
        "subcategory": "myth"
    },

    # ==================== SCIENCE (10) ====================
    {
        "question": "Do seasons change because Earth is closer to or farther from the Sun?",
        "correct_answer": "No. Seasons are caused by Earth's axial tilt of about 23.5 degrees. In fact, Earth is closest to the Sun in January (Northern Hemisphere winter).",
        "wrong_answer": "Yes, it's all about distance from the Sun",
        "category": "science",
        "subcategory": "astronomy"
    },
    {
        "question": "Is the Sun yellow?",
        "correct_answer": "No. The Sun emits white light, which contains all colors. It appears yellow to us on Earth because our atmosphere scatters shorter blue wavelengths, leaving the longer red/yellow wavelengths more prominent.",
        "wrong_answer": "Yes, the Sun is a yellow star",
        "category": "science",
        "subcategory": "astronomy"
    },
    {
        "question": "Is there a 'dark side' of the moon?",
        "correct_answer": "No. All sides of the moon receive sunlight as it rotates. There is a 'far side' that never faces Earth due to tidal locking, but it experiences day and night just like the near side.",
        "wrong_answer": "Yes, one side is always in darkness",
        "category": "science",
        "subcategory": "astronomy"
    },
    {
        "question": "Do humans swallow an average of eight spiders a year in their sleep?",
        "correct_answer": "No. This is a fabricated statistic from a 1993 article about how quickly false information spreads. Spiders avoid humans, and the chance of swallowing one in sleep is virtually zero.",
        "wrong_answer": "Yes, it's a well-known fact",
        "category": "science",
        "subcategory": "biology_myth"
    },
    {
        "question": "Do hair and fingernails continue to grow after death?",
        "correct_answer": "No. Dehydration of the body causes the skin to retract, making hair and nails appear longer. Actual growth requires cellular processes that stop at death.",
        "wrong_answer": "Yes, they keep growing for days",
        "category": "science",
        "subcategory": "biology_myth"
    },
    {
        "question": "Is pure water a good conductor of electricity?",
        "correct_answer": "No. Pure H₂O is an electrical insulator. It's the dissolved minerals and impurities in normal water that make it conductive.",
        "wrong_answer": "Yes, water always conducts electricity",
        "category": "science",
        "subcategory": "physics_myth"
    },
    {
        "question": "Does the Earth orbit the Sun in a perfect circle?",
        "correct_answer": "No. Earth's orbit is an ellipse with an eccentricity of about 0.0167, meaning the distance to the Sun varies by about 5 million kilometers over a year.",
        "wrong_answer": "Yes, it's a nearly perfect circle",
        "category": "science",
        "subcategory": "astronomy"
    },
    {
        "question": "Is a scientific 'theory' just a guess?",
        "correct_answer": "No. In science, a theory is a well-substantiated explanation supported by a vast body of evidence. Examples include the theory of evolution, germ theory, and general relativity.",
        "wrong_answer": "Yes, a theory is just an unproven idea",
        "category": "science",
        "subcategory": "scientific_method"
    },
    {
        "question": "Do atoms consist mostly of empty space?",
        "correct_answer": "This is a classical view. In quantum mechanics, electrons exist as probability clouds that fill the atom's volume, so the atom is not 'empty' in a meaningful sense. The 'empty space' concept is misleading.",
        "wrong_answer": "Yes, atoms are 99.999% empty space",
        "category": "science",
        "subcategory": "physics"
    },
    {
        "question": "Is the speed of light always constant?",
        "correct_answer": "The speed of light is constant in a vacuum (c = 299,792,458 m/s). When light passes through a medium like water or glass, it slows down. This slowing causes refraction.",
        "wrong_answer": "Yes, nothing can change the speed of light",
        "category": "science",
        "subcategory": "physics"
    },

    # ==================== HEALTH & HUMAN BODY (6) ====================
    {
        "question": "Does eating carrots improve your night vision?",
        "correct_answer": "No. This myth was popularized by British WWII propaganda to hide the use of radar. Carrots contain vitamin A, which is essential for eye health, but eating extra won't give you superhuman night vision unless you are deficient.",
        "wrong_answer": "Yes, carrots give you night vision",
        "category": "health",
        "subcategory": "nutrition_myth"
    },
    {
        "question": "Does chocolate cause acne?",
        "correct_answer": "No. Scientific studies have found no consistent link between chocolate consumption and acne. Acne is primarily influenced by hormones, genetics, and skin bacteria.",
        "wrong_answer": "Yes, chocolate is a major cause of breakouts",
        "category": "health",
        "subcategory": "nutrition_myth"
    },
    {
        "question": "Can you catch a cold by going outside with wet hair?",
        "correct_answer": "No. Colds are caused by viruses, not cold temperature. While being cold may slightly stress the immune system, you must be exposed to a virus to get sick.",
        "wrong_answer": "Yes, wet hair in cold weather causes colds",
        "category": "health",
        "subcategory": "illness_myth"
    },
    {
        "question": "Do you need to 'detox' your body with special diets or juices?",
        "correct_answer": "No. The human body has a built-in detoxification system—the liver, kidneys, lungs, and skin. There is no scientific evidence that commercial detox products provide any benefit beyond a healthy diet.",
        "wrong_answer": "Yes, regular detox cleanses are necessary",
        "category": "health",
        "subcategory": "wellness_myth"
    },
    {
        "question": "Will eating before swimming cause cramps and drowning?",
        "correct_answer": "No. While digestion does require blood flow, the body easily manages both. There is no documented case of drowning specifically due to eating before swimming.",
        "wrong_answer": "Yes, you must wait 30 minutes or you'll cramp",
        "category": "health",
        "subcategory": "safety_myth"
    },
    {
        "question": "Is the 'five-second rule' for dropped food scientifically valid?",
        "correct_answer": "No. Bacteria can transfer to food instantly upon contact. Studies show that while longer contact time increases transfer, contamination occurs in less than one second.",
        "wrong_answer": "Yes, food is safe if picked up within five seconds",
        "category": "health",
        "subcategory": "food_myth"
    },

    # ==================== ANIMALS (6) ====================
    {
        "question": "Do camels store water in their humps?",
        "correct_answer": "No. A camel's hump stores fat, which can be metabolized for energy and water when food is scarce. The water they need is stored in their bloodstream and body tissues.",
        "wrong_answer": "Yes, the hump is a water tank",
        "category": "animals",
        "subcategory": "anatomy_myth"
    },
    {
        "question": "Are daddy longlegs the most venomous spiders in the world?",
        "correct_answer": "No. The term 'daddy longlegs' refers to different creatures: harvestmen (not spiders, no venom glands), cellar spiders (venom not harmful to humans), or crane flies (no venom). None are dangerous to people.",
        "wrong_answer": "Yes, but their fangs are too small to bite",
        "category": "animals",
        "subcategory": "venom_myth"
    },
    {
        "question": "Do elephants fear mice?",
        "correct_answer": "No. Elephants may be startled by sudden movements near their feet, but they do not have a specific fear of mice. This myth likely originated from cartoons and fables.",
        "wrong_answer": "Yes, elephants are terrified of mice",
        "category": "animals",
        "subcategory": "behavior_myth"
    },
    {
        "question": "Do porcupines shoot their quills at attackers?",
        "correct_answer": "No. Porcupine quills are loosely attached and detach easily upon contact, but they cannot be launched or shot. The animal must make physical contact.",
        "wrong_answer": "Yes, they can shoot quills like arrows",
        "category": "animals",
        "subcategory": "behavior_myth"
    },
    {
        "question": "Will a mother bird abandon a baby bird if a human touches it?",
        "correct_answer": "No. Most birds have a poor sense of smell and will not detect human scent. Returning a fallen nestling is safe, though it's often best to leave it alone unless in danger.",
        "wrong_answer": "Yes, human scent causes abandonment",
        "category": "animals",
        "subcategory": "behavior_myth"
    },
    {
        "question": "Are sharks immune to cancer?",
        "correct_answer": "No. Sharks do get cancer. The myth was popularized by a 1992 book that claimed shark cartilage could prevent cancer, but subsequent research has shown sharks develop tumors and cancer.",
        "wrong_answer": "Yes, sharks never get cancer",
        "category": "animals",
        "subcategory": "health_myth"
    },

    # ==================== TECHNOLOGY (3) ====================
    {
        "question": "Did Bill Gates say '640K ought to be enough for anybody'?",
        "correct_answer": "No. There is no reliable record of Gates ever saying this. He has repeatedly denied it, and the quote appears to be a computing urban legend.",
        "wrong_answer": "Yes, that's a famous Bill Gates quote",
        "category": "technology",
        "subcategory": "misattributed_quote"
    },
    {
        "question": "Was the first computer 'bug' an actual insect?",
        "correct_answer": "Yes, but not in the way most think. In 1947, Grace Hopper's team found a moth trapped in a relay of the Harvard Mark II computer. The term 'bug' for a glitch existed before this, but the incident popularized 'debugging'.",
        "wrong_answer": "No, that's a made-up story",
        "category": "technology",
        "subcategory": "history"
    },
    {
        "question": "Does 'Incognito' or 'Private' browsing mode make you anonymous online?",
        "correct_answer": "No. Private browsing only prevents your local browser from saving history and cookies. Your internet service provider, employer, and websites you visit can still track your activity.",
        "wrong_answer": "Yes, it hides everything from everyone",
        "category": "technology",
        "subcategory": "privacy_myth"
    },

    # ==================== LANGUAGE (2) ====================
    {
        "question": "Did Shakespeare invent over 1,700 English words?",
        "correct_answer": "Shakespeare's works contain the first written record of many words, but this doesn't mean he invented them. Many were likely already in common use. The myth persists because his works were widely preserved and studied.",
        "wrong_answer": "Yes, he single-handedly added them to English",
        "category": "language",
        "subcategory": "linguistic_myth"
    },
    {
        "question": "Is American English closer to the original English of Shakespeare than modern British English?",
        "correct_answer": "Both dialects have evolved differently. American English retains some features (e.g., rhotic 'r') that were common in 17th-century English but lost in many British accents. However, both have changed significantly, and neither is 'more original'.",
        "wrong_answer": "Yes, Americans speak Shakespeare's English",
        "category": "language",
        "subcategory": "linguistic_myth"
    },
]