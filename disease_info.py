"""
Disease Information Database
============================
Contains detailed descriptions, symptoms, causes, treatments, prevention tips,
severity levels, scientific names, and recommended actions for all 38 plant
disease classes in the PlantVillage dataset.

Enhanced for the modernized PyTorch-based Plant Disease Detection project.
"""

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "scientific_name": "Venturia inaequalis",
        "description": "Apple scab is a common fungal disease affecting apple trees, causing lesions on leaves and fruit.",
        "symptoms": "Dark, olive-green to black, roughly circular lesions on leaves and fruit. Leaves may twist and drop prematurely. Fruit may crack.",
        "causes": "Fungus: Venturia inaequalis. Spreads via wind-blown ascospores during wet spring weather.",
        "treatment": "Apply fungicides such as captan or myclobutanil starting at green tip stage. Prune infected shoots.",
        "prevention": "Rake and destroy fallen leaves in autumn. Plant scab-resistant cultivars (e.g., Liberty, Enterprise). Ensure good air circulation.",
        "severity": "Medium",
        "recommended_actions": "Begin fungicide program at green tip. Monitor weather for infection periods. Remove fallen leaves in autumn."
    },
    "Apple___Black_rot": {
        "name": "Apple Black Rot",
        "scientific_name": "Botryosphaeria obtusa",
        "description": "Black rot is a widespread fungal disease that affects the fruit, leaves, and bark of apple trees.",
        "symptoms": "'Frog-eye' leaf spots (brown with purple margins). Expanding brown to black rot on fruit leading to mummification. Bark cankers.",
        "causes": "Fungus: Botryosphaeria obtusa. Overwinters in dead wood and mummified fruit.",
        "treatment": "Prune out and burn all cankers and dead wood. Remove mummified fruit. Apply captan or thiophanate-methyl.",
        "prevention": "Maintain tree vigor. Remove pruning debris from the orchard. Avoid mechanical damage to trees.",
        "severity": "High",
        "recommended_actions": "Remove all mummified fruit and dead wood immediately. Apply protective fungicides during wet periods."
    },
    "Apple___Cedar_apple_rust": {
        "name": "Cedar Apple Rust",
        "scientific_name": "Gymnosporangium juniperi-virginianae",
        "description": "A rust disease that requires two hosts (apple and cedar/juniper) to complete its life cycle.",
        "symptoms": "Bright orange-yellow spots on upper leaf surfaces. Orange tubular structures on undersides. Premature defoliation.",
        "causes": "Fungus: Gymnosporangium juniperi-virginianae. Spores blow from cedar galls to apple trees during wet spring weather.",
        "treatment": "Apply fungicide sprays containing myclobutanil or triadimefon during the infection period.",
        "prevention": "Remove nearby Eastern red cedar/juniper hosts if possible. Plant rust-resistant apple varieties.",
        "severity": "Medium",
        "recommended_actions": "Remove cedar/juniper trees within a few hundred feet if feasible. Apply fungicides during wet spring periods."
    },
    "Apple___healthy": {
        "name": "Apple (Healthy)",
        "scientific_name": "Malus domestica",
        "description": "The apple plant appears completely healthy.",
        "symptoms": "None. Leaves are green and unblemished.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Continue standard orchard management: proper pruning, balanced fertilization, and routine scouting.",
        "severity": "None",
        "recommended_actions": "Continue regular maintenance. Monitor for early signs of disease during humid weather."
    },
    "Blueberry___healthy": {
        "name": "Blueberry (Healthy)",
        "scientific_name": "Vaccinium corymbosum",
        "description": "The blueberry plant appears healthy with no signs of disease.",
        "symptoms": "None.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Maintain acidic soil pH (4.5\u20135.5), provide adequate mulching, and ensure proper drainage.",
        "severity": "None",
        "recommended_actions": "Continue maintaining acidic soil conditions and proper irrigation."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "name": "Cherry Powdery Mildew",
        "scientific_name": "Podosphaera clandestina",
        "description": "A common fungal disease causing a white powdery coating on cherry plant surfaces.",
        "symptoms": "White, powdery fungal growth on young leaves, shoots, and sometimes fruit. Leaves may curl and distort.",
        "causes": "Fungus: Podosphaera clandestina. Thrives in warm, dry weather with high relative humidity.",
        "treatment": "Apply sulfur-based fungicides, potassium bicarbonate, or myclobutanil at the first sign of infection.",
        "prevention": "Prune to ensure good air circulation. Avoid excessive nitrogen fertilization. Remove infected plant debris.",
        "severity": "Medium",
        "recommended_actions": "Improve air circulation through pruning. Apply sulfur-based fungicide at first sign of white patches."
    },
    "Cherry_(including_sour)___healthy": {
        "name": "Cherry (Healthy)",
        "scientific_name": "Prunus avium / Prunus cerasus",
        "description": "The cherry plant appears healthy with no signs of disease.",
        "symptoms": "None.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper pruning, ensure good drainage, and apply preventive fungicide during humid periods.",
        "severity": "None",
        "recommended_actions": "Continue standard care practices. Scout regularly for early signs of disease."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "name": "Corn Gray Leaf Spot",
        "scientific_name": "Cercospora zeae-maydis",
        "description": "A major foliar disease of corn that can significantly reduce yields by destroying leaf tissue.",
        "symptoms": "Small tan spots that elongate into distinct rectangular, gray-to-tan lesions restricted by leaf veins.",
        "causes": "Fungus: Cercospora zeae-maydis. Survives in corn residue on the soil surface. Favored by high humidity.",
        "treatment": "Apply foliar fungicides containing strobilurin or triazole at the VT/R1 (tasseling) growth stage.",
        "prevention": "Plant resistant corn hybrids. Practice crop rotation and tillage to bury infected crop residue.",
        "severity": "High",
        "recommended_actions": "Apply fungicide at tasseling stage if conditions favor disease. Rotate crops next season."
    },
    "Corn_(maize)___Common_rust_": {
        "name": "Corn Common Rust",
        "scientific_name": "Puccinia sorghi",
        "description": "A fungal disease characterized by rust-colored pustules on corn leaves.",
        "symptoms": "Small, circular to elongate, cinnamon-brown powdery pustules on both upper and lower leaf surfaces.",
        "causes": "Fungus: Puccinia sorghi. Spores are carried by wind from southern climates. Thrives in cool, moist weather.",
        "treatment": "Foliar fungicides (mancozeb, propiconazole) can be applied if infection is severe before tasseling.",
        "prevention": "Plant rust-resistant hybrids. Early planting can help avoid peak spore dispersal periods in mid-summer.",
        "severity": "Medium",
        "recommended_actions": "Monitor pustule density. Apply foliar fungicide if severe infection occurs before tasseling."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "name": "Corn Northern Leaf Blight",
        "scientific_name": "Exserohilum turcicum",
        "description": "A foliar disease causing large lesions that can cause major yield loss if established before silking.",
        "symptoms": "Long (1-6 inches), cigar-shaped, grayish-green to tan lesions on lower leaves, spreading upwards.",
        "causes": "Fungus: Exserohilum turcicum. Overwinters in corn residue. Favored by moderate temperatures and heavy dew.",
        "treatment": "Apply fungicides (azoxystrobin, propiconazole) at or before tasseling when lesions are first observed.",
        "prevention": "Use resistant hybrids (containing Ht genes). Practice 1-2 year crop rotation and tillage to bury residue.",
        "severity": "High",
        "recommended_actions": "Apply fungicide immediately at first sign of cigar-shaped lesions. Plan crop rotation for next season."
    },
    "Corn_(maize)___healthy": {
        "name": "Corn (Healthy)",
        "scientific_name": "Zea mays",
        "description": "The corn plant appears healthy with no visible signs of disease.",
        "symptoms": "None.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Use certified seed, maintain proper plant spacing, rotate crops, and scout regularly.",
        "severity": "None",
        "recommended_actions": "Continue standard crop management and regular scouting."
    },
    "Grape___Black_rot": {
        "name": "Grape Black Rot",
        "scientific_name": "Guignardia bidwellii",
        "description": "A devastating fungal disease that can destroy an entire grape crop if left unmanaged.",
        "symptoms": "Tan, circular lesions on leaves with dark borders. Berries shrivel into hard, black, raisin-like mummies.",
        "causes": "Fungus: Guignardia bidwellii. Overwinters in mummified berries and cane lesions. Favored by warm, wet weather.",
        "treatment": "Apply fungicides (myclobutanil, mancozeb) from early growth stages through veraison.",
        "prevention": "Prune out and destroy all mummified fruit and infected canes during dormancy. Manage canopy for airflow.",
        "severity": "Critical",
        "recommended_actions": "Remove all mummified berries immediately. Begin aggressive fungicide program. Improve canopy airflow."
    },
    "Grape___Esca_(Black_Measles)": {
        "name": "Grape Esca (Black Measles)",
        "scientific_name": "Phaeomoniella chlamydospora / Phaeoacremonium spp.",
        "description": "A complex fungal trunk disease that slowly degrades the vascular tissue of grapevines.",
        "symptoms": "Interveinal yellowing/browning ('tiger stripe' pattern) on leaves. Dark, spotting ('measles') on berries.",
        "causes": "Complex of fungi (Phaeomoniella, Phaeoacremonium). Infection usually occurs through pruning wounds.",
        "treatment": "No fully effective chemical cure. Remedial surgery to remove infected wood. Apply wound protectants after pruning.",
        "prevention": "Avoid pruning during wet weather. Treat large pruning wounds with fungicide paste. Use clean nursery stock.",
        "severity": "Critical",
        "recommended_actions": "Protect pruning wounds immediately. Consider remedial trunk surgery. Replace severely affected vines."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "name": "Grape Leaf Blight",
        "scientific_name": "Pseudocercospora vitis",
        "description": "A late-season foliar disease causing premature defoliation and weakening of the vine.",
        "symptoms": "Irregular, dark red to brown lesions on leaves. In severe cases, leaves dry up and drop prematurely.",
        "causes": "Fungus: Pseudocercospora vitis. Overwinters in fallen leaves. Thrives in high humidity and rainfall.",
        "treatment": "Apply copper-based fungicides or mancozeb at regular intervals during the growing season.",
        "prevention": "Collect and destroy fallen leaves. Ensure proper canopy management to improve airflow and reduce humidity.",
        "severity": "Medium",
        "recommended_actions": "Apply copper-based fungicide. Improve canopy management. Remove and destroy fallen leaves."
    },
    "Grape___healthy": {
        "name": "Grape (Healthy)",
        "scientific_name": "Vitis vinifera",
        "description": "The grape plant appears completely healthy.",
        "symptoms": "None.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper canopy management, balanced nutrition, and standard preventive fungicide programs.",
        "severity": "None",
        "recommended_actions": "Continue standard vineyard management practices."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "name": "Citrus Greening (HLB)",
        "scientific_name": "Candidatus Liberibacter asiaticus",
        "description": "A deadly, incurable bacterial disease that severely impacts citrus production worldwide.",
        "symptoms": "Asymmetrical, blotchy mottling of leaves. Fruit is lopsided, stays green at the bottom, and tastes bitter.",
        "causes": "Bacterium: Candidatus Liberibacter asiaticus. Spread by the Asian citrus psyllid (an insect vector).",
        "treatment": "No cure exists. Infected trees must be removed and destroyed to protect surrounding trees.",
        "prevention": "Control the psyllid vector with systemic insecticides. Use certified disease-free nursery stock. Quarantine areas.",
        "severity": "Critical",
        "recommended_actions": "Remove and destroy infected trees immediately. Implement psyllid control in surrounding areas."
    },
    "Peach___Bacterial_spot": {
        "name": "Peach Bacterial Spot",
        "scientific_name": "Xanthomonas arboricola pv. pruni",
        "description": "A bacterial disease affecting the foliage and fruit of stone fruits, causing cosmetic damage and defoliation.",
        "symptoms": "Small, angular, water-soaked leaf spots that turn brown and drop out ('shot-hole' effect). Pitted spots on fruit.",
        "causes": "Bacterium: Xanthomonas arboricola pv. pruni. Overwinters in twig cankers and buds. Spread by rain splash.",
        "treatment": "Apply copper-based bactericides or oxytetracycline during the early growing season.",
        "prevention": "Plant resistant peach varieties. Avoid planting near susceptible plum varieties. Maintain balanced tree nutrition.",
        "severity": "Medium",
        "recommended_actions": "Apply copper bactericide during early growing season. Consider planting resistant varieties."
    },
    "Peach___healthy": {
        "name": "Peach (Healthy)",
        "scientific_name": "Prunus persica",
        "description": "The peach plant appears healthy with no visible signs of disease.",
        "symptoms": "None.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper pruning, sanitation, and apply dormant copper sprays to prevent future infections.",
        "severity": "None",
        "recommended_actions": "Continue regular orchard management and dormant-season copper applications."
    },
    "Pepper,_bell___Bacterial_spot": {
        "name": "Bell Pepper Bacterial Spot",
        "scientific_name": "Xanthomonas campestris pv. vesicatoria",
        "description": "A severe bacterial disease of peppers that can lead to total crop loss under favorable conditions.",
        "symptoms": "Small, water-soaked spots on leaves that turn necrotic. Leaves yellow and drop. Raised, scab-like spots on fruit.",
        "causes": "Bacterium: Xanthomonas campestris pv. vesicatoria. Seed-borne and spread by splashing rain or overhead irrigation.",
        "treatment": "Apply copper bactericides combined with mancozeb. Plant defense elicitors (e.g., Actigard) can help.",
        "prevention": "Use certified disease-free seeds and transplants. Practice 3-year crop rotation. Use drip irrigation instead of overhead.",
        "severity": "High",
        "recommended_actions": "Switch to drip irrigation immediately. Apply copper + mancozeb. Plan 3-year rotation."
    },
    "Pepper,_bell___healthy": {
        "name": "Bell Pepper (Healthy)",
        "scientific_name": "Capsicum annuum",
        "description": "The bell pepper plant appears completely healthy.",
        "symptoms": "None.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Use disease-free transplants, rotate crops, maintain proper spacing, and water at the base of plants.",
        "severity": "None",
        "recommended_actions": "Continue standard care. Water at base of plants to prevent foliar diseases."
    },
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "scientific_name": "Alternaria solani",
        "description": "A common fungal disease of potatoes that primarily affects older foliage.",
        "symptoms": "Dark brown to black lesions on older leaves, featuring distinct concentric rings ('target' appearance).",
        "causes": "Fungus: Alternaria solani. Survives in infected plant debris in the soil. Favored by alternating wet and dry conditions.",
        "treatment": "Apply protectant fungicides (chlorothalonil, mancozeb) starting when symptoms first appear.",
        "prevention": "Practice 2-3 year crop rotation. Ensure plants are adequately fertilized to prevent premature aging and stress.",
        "severity": "Medium",
        "recommended_actions": "Begin protectant fungicide spray program. Ensure adequate plant nutrition."
    },
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "scientific_name": "Phytophthora infestans",
        "description": "A highly destructive disease (famous for the Irish Potato Famine) that can destroy a field in days.",
        "symptoms": "Large, irregular, water-soaked lesions on leaves turning brown. White, fuzzy mold on the underside of leaves. Tuber rot.",
        "causes": "Oomycete: Phytophthora infestans. Spreads rapidly in cool, wet, and humid weather via wind-blown spores.",
        "treatment": "Apply specific systemic fungicides (e.g., metalaxyl, cymoxanil) immediately upon detection. Destroy severely infected plants.",
        "prevention": "Use certified disease-free seed potatoes. Eliminate cull piles and volunteer potatoes. Monitor weather forecasting models.",
        "severity": "Critical",
        "recommended_actions": "Apply systemic fungicide IMMEDIATELY. Destroy severely infected plants. Monitor neighboring fields."
    },
    "Potato___healthy": {
        "name": "Potato (Healthy)",
        "scientific_name": "Solanum tuberosum",
        "description": "The potato plant appears healthy with no visible signs of disease.",
        "symptoms": "None.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Use certified seed, rotate crops, maintain proper hilling, and scout regularly for early disease symptoms.",
        "severity": "None",
        "recommended_actions": "Continue standard crop management and regular scouting."
    },
    "Raspberry___healthy": {
        "name": "Raspberry (Healthy)",
        "scientific_name": "Rubus idaeus",
        "description": "The raspberry plant appears completely healthy.",
        "symptoms": "None.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper cane spacing, remove spent floricanes after harvest, and ensure good soil drainage.",
        "severity": "None",
        "recommended_actions": "Continue removing spent canes and maintaining proper spacing."
    },
    "Soybean___healthy": {
        "name": "Soybean (Healthy)",
        "scientific_name": "Glycine max",
        "description": "The soybean plant appears completely healthy.",
        "symptoms": "None.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Rotate crops with non-legumes, use fungicide-treated seeds, and control weed populations.",
        "severity": "None",
        "recommended_actions": "Continue crop rotation and regular scouting."
    },
    "Squash___Powdery_mildew": {
        "name": "Squash Powdery Mildew",
        "scientific_name": "Podosphaera xanthii",
        "description": "A prevalent fungal disease on cucurbits that reduces photosynthesis and yield.",
        "symptoms": "White, powdery fungal spots on the upper and lower surfaces of leaves and stems. Leaves may turn yellow and wither.",
        "causes": "Fungi: Podosphaera xanthii or Erysiphe cichoracearum. Thrives in warm, dry climates with high humidity at plant level.",
        "treatment": "Apply fungicides (potassium bicarbonate, sulfur, or specific systemic fungicides) at the first sign of mildew.",
        "prevention": "Plant powdery mildew-resistant squash varieties. Ensure adequate plant spacing for airflow. Avoid excess nitrogen.",
        "severity": "Medium",
        "recommended_actions": "Apply potassium bicarbonate or sulfur spray at first sign. Improve plant spacing."
    },
    "Strawberry___Leaf_scorch": {
        "name": "Strawberry Leaf Scorch",
        "scientific_name": "Diplocarpon earlianum",
        "description": "A fungal disease affecting strawberry foliage that can severely reduce plant vigor.",
        "symptoms": "Numerous small, irregular, dark purple spots on upper leaf surfaces. Leaves may curl, turn brown, and look 'scorched'.",
        "causes": "Fungus: Diplocarpon earlianum. Overwinters in dead leaves. Splashing rain spreads the spores.",
        "treatment": "Apply fungicides (captan, myclobutanil) beginning at bloom and continuing through harvest.",
        "prevention": "Use disease-resistant varieties. Renovate strawberry beds promptly after harvest. Avoid overhead irrigation.",
        "severity": "Medium",
        "recommended_actions": "Begin fungicide applications at bloom. Renovate beds after harvest. Switch to drip irrigation."
    },
    "Strawberry___healthy": {
        "name": "Strawberry (Healthy)",
        "scientific_name": "Fragaria × ananassa",
        "description": "The strawberry plant appears completely healthy.",
        "symptoms": "None.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Plant in well-drained soil, ensure good spacing, and renovate beds annually to maintain health.",
        "severity": "None",
        "recommended_actions": "Continue standard bed maintenance and annual renovation."
    },
    "Tomato___Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "scientific_name": "Xanthomonas vesicatoria",
        "description": "A severe bacterial disease of tomatoes causing spotting on leaves and rendering fruit unmarketable.",
        "symptoms": "Small, dark, water-soaked spots on leaves. Leaves turn yellow and drop. Small, raised, blister-like spots on green fruit.",
        "causes": "Bacterium: Xanthomonas species. Spread by seed, infected transplants, splashing rain, and working with wet plants.",
        "treatment": "Apply fixed copper sprays combined with mancozeb to slow the spread. No chemical completely cures the disease.",
        "prevention": "Use certified disease-free seeds/transplants. Practice 3-year crop rotation. Use drip irrigation. Do not work wet fields.",
        "severity": "High",
        "recommended_actions": "Apply copper + mancozeb immediately. Switch to drip irrigation. Avoid working with wet plants."
    },
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "scientific_name": "Alternaria solani",
        "description": "A common fungal disease causing defoliation, starting from the bottom of the tomato plant.",
        "symptoms": "Dark brown spots with concentric rings (target appearance) on older, lower leaves first. Leaves turn yellow and drop.",
        "causes": "Fungus: Alternaria solani. Survives in soil and plant debris. Favored by warm, wet, and humid conditions.",
        "treatment": "Apply protectant fungicides (chlorothalonil, mancozeb) on a 7-10 day schedule once symptoms appear.",
        "prevention": "Rotate away from solanaceous crops. Stake/cage plants to improve airflow. Mulch to prevent soil splashing onto lower leaves.",
        "severity": "Medium",
        "recommended_actions": "Begin 7-10 day fungicide spray schedule. Mulch around base of plants to reduce soil splash."
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "scientific_name": "Phytophthora infestans",
        "description": "A devastating disease that can rapidly destroy entire tomato crops under cool, wet conditions.",
        "symptoms": "Irregular, water-soaked green-black spots on leaves. White fuzzy mold on leaf undersides. Firm, brown rot on fruit.",
        "causes": "Oomycete: Phytophthora infestans. Wind-borne spores spread rapidly in cool, wet weather.",
        "treatment": "Apply specific fungicides (chlorothalonil, cymoxanil) immediately. Destroy severely infected plants by bagging them.",
        "prevention": "Plant resistant varieties (e.g., Mountain Magic). Keep foliage dry. Destroy volunteer potatoes and tomatoes nearby.",
        "severity": "Critical",
        "recommended_actions": "Apply fungicide IMMEDIATELY upon detection. Remove and bag severely infected plants. Alert neighboring growers."
    },
    "Tomato___Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "scientific_name": "Passalora fulva",
        "description": "A fungal disease primarily affecting tomatoes grown in greenhouses or high tunnels with poor ventilation.",
        "symptoms": "Pale green/yellow spots on the upper leaf surface. Olive-green to brown velvety mold patches on the underside.",
        "causes": "Fungus: Passalora fulva. Highly dependent on high relative humidity (>85%) and poor air circulation.",
        "treatment": "Apply chlorothalonil or copper-based fungicides. Dramatically increasing ventilation is critical for control.",
        "prevention": "Grow resistant varieties. Maximize greenhouse ventilation. Space plants properly and prune lower leaves.",
        "severity": "Medium",
        "recommended_actions": "Increase ventilation immediately. Apply chlorothalonil. Prune lower leaves for airflow."
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "scientific_name": "Septoria lycopersici",
        "description": "A highly destructive foliar disease that causes rapid defoliation but does not directly infect the fruit.",
        "symptoms": "Numerous small, circular spots with dark borders and tan/gray centers. Tiny black specks (pycnidia) in the center of spots.",
        "causes": "Fungus: Septoria lycopersici. Overwinters on tomato debris and solanaceous weeds. Spread by splashing water.",
        "treatment": "Apply fungicides (chlorothalonil, mancozeb) on a 7-10 day schedule, especially during rainy periods.",
        "prevention": "Remove and destroy all tomato debris at season end. Practice a 2-3 year crop rotation. Use mulch to reduce splashing.",
        "severity": "High",
        "recommended_actions": "Begin 7-10 day fungicide schedule immediately. Remove infected lower leaves. Apply mulch."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Two-Spotted Spider Mite",
        "scientific_name": "Tetranychus urticae",
        "description": "Microscopic arachnids that suck sap from plant cells, causing severe stress and defoliation.",
        "symptoms": "Tiny yellow or white speckles (stippling) on leaves. Fine silken webbing on the undersides of leaves. Leaves turn bronze and dry up.",
        "causes": "Pest: Tetranychus urticae. Populations explode during hot, dry, and dusty weather.",
        "treatment": "Apply specific miticides (e.g., abamectin) or horticultural oils/soaps. Ensure thorough coverage of leaf undersides.",
        "prevention": "Avoid plant drought stress by proper watering. Release predatory mites (Phytoseiulus persimilis). Control dust around plants.",
        "severity": "High",
        "recommended_actions": "Apply miticide with thorough coverage of leaf undersides. Increase watering to reduce plant stress."
    },
    "Tomato___Target_Spot": {
        "name": "Tomato Target Spot",
        "scientific_name": "Corynespora cassiicola",
        "description": "A fungal disease causing lesions on leaves and deep pitting on tomato fruit.",
        "symptoms": "Small, brown lesions with concentric rings on leaves, resembling early blight. Deep, pitted, dark spots on ripe fruit.",
        "causes": "Fungus: Corynespora cassiicola. Thrives in high humidity and temperatures between 68-82\u00b0F.",
        "treatment": "Apply systemic fungicides (azoxystrobin, difenoconazole) or protectants (chlorothalonil).",
        "prevention": "Ensure good air circulation via pruning and staking. Avoid prolonged leaf wetness. Remove infected debris.",
        "severity": "Medium",
        "recommended_actions": "Apply systemic fungicide. Improve air circulation through pruning and staking."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "scientific_name": "Tomato yellow leaf curl virus (TYLCV)",
        "description": "A highly damaging viral disease that severely stunts plants and stops fruit production.",
        "symptoms": "Severe stunting of the plant. Leaves curl upwards and turn distinctly yellow at the margins. Flowers drop off.",
        "causes": "Virus: TYLCV. Transmitted exclusively by the silverleaf whitefly (Bemisia tabaci).",
        "treatment": "There is no cure for the virus. Infected plants must be immediately uprooted and destroyed to prevent spread.",
        "prevention": "Plant TYLCV-resistant varieties. Use fine mesh netting in greenhouses to exclude whiteflies. Control whitefly populations.",
        "severity": "Critical",
        "recommended_actions": "Remove and destroy infected plants IMMEDIATELY. Implement aggressive whitefly control program."
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus (ToMV)",
        "scientific_name": "Tomato mosaic virus (ToMV)",
        "description": "A highly contagious viral disease that affects plant vigor and fruit quality.",
        "symptoms": "Light and dark green mottled patterns (mosaic) on leaves. Leaves may be fern-like or distorted. Internal browning of fruit.",
        "causes": "Virus: ToMV. Extremely stable virus spread mechanically by hands, tools, clothing, and seed. Not spread by insects.",
        "treatment": "No cure exists. Remove and destroy infected plants immediately. Do not compost them.",
        "prevention": "Grow resistant varieties (check for 'T' in the resistance code). Wash hands with soap and water. Disinfect tools.",
        "severity": "High",
        "recommended_actions": "Remove infected plants immediately. Disinfect all tools and wash hands before touching other plants."
    },
    "Tomato___healthy": {
        "name": "Tomato (Healthy)",
        "scientific_name": "Solanum lycopersicum",
        "description": "The tomato plant appears completely healthy.",
        "symptoms": "None.",
        "causes": "N/A",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper spacing, stake or cage plants, apply mulch, and water at the base to prevent future issues.",
        "severity": "None",
        "recommended_actions": "Continue standard care. Water at plant base and maintain proper staking."
    }
}

def get_disease_info(class_name):
    """
    Returns a dictionary with comprehensive information for the given class name.
    Falls back to generic info if the class is not in the database.

    Keys returned: name, scientific_name, description, symptoms, causes,
    treatment, prevention, severity, recommended_actions.
    """
    return DISEASE_INFO.get(class_name, {
        'name': class_name.replace('___', ' - ').replace('_', ' '),
        'scientific_name': 'Unknown',
        'description': f"Condition identified as '{class_name}'. Detailed information is not available.",
        'symptoms': "Symptoms unknown for this specific classification.",
        'causes': "Causes unknown.",
        'treatment': "Consult your local agricultural extension office for accurate treatment protocols.",
        'prevention': "Maintain good agricultural practices, rotate crops, and sanitize equipment.",
        'severity': "Unknown",
        'recommended_actions': "Consult a local agricultural expert for specific recommendations."
    })
