// ============================================
// CONFIGURATION & AI MODEL
// ============================================

class FitnessAI {
    constructor() {
        this.model = null;
        this.isTrained = false;
        // Synthetic data for training "on the fly" to simulate learning
        this.trainingData = {
            // Inputs: [Weight (kg), Height (cm), Activity Level (1-5)]
            inputs: tf.tensor2d([
                [50, 160, 1], [60, 165, 2], [70, 170, 3], [80, 175, 4], [90, 180, 5],
                [55, 155, 1], [65, 160, 2], [75, 175, 3], [85, 180, 4], [95, 185, 5],
                [100, 170, 1], [110, 175, 1]
            ]),
            // Outputs: [Calories, Protein (g), Workout Intensity (1-10)]
            outputs: tf.tensor2d([
                [1500, 60, 2], [1800, 80, 4], [2200, 120, 6], [2600, 150, 8], [3000, 180, 10],
                [1400, 55, 2], [1700, 75, 4], [2100, 110, 6], [2500, 140, 8], [2900, 170, 10],
                [2000, 90, 2], [2200, 100, 2]
            ])
        };
    }

    async createModel() {
        const model = tf.sequential();

        // Input Layer
        model.add(tf.layers.dense({
            inputShape: [3],
            units: 16,
            activation: 'relu'
        }));

        // Hidden Layer
        model.add(tf.layers.dense({
            units: 8,
            activation: 'relu'
        }));

        // Output Layer (Calories, Protein, Intensity)
        model.add(tf.layers.dense({
            units: 3,
            activation: 'linear'
        }));

        model.compile({
            optimizer: tf.train.adam(0.01),
            loss: 'meanSquaredError'
        });

        this.model = model;
        return model;
    }

    async train(progressCallback) {
        if (!this.model) await this.createModel();

        // Train the model
        return await this.model.fit(this.trainingData.inputs, this.trainingData.outputs, {
            epochs: 50,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const progress = Math.round(((epoch + 1) / 50) * 100);
                    if (progressCallback) progressCallback(progress, logs.loss);
                }
            }
        });
    }

    predict(weight, height, activityLevel) {
        const input = tf.tensor2d([[weight, height, this.mapActivityLevel(activityLevel)]]);
        const prediction = this.model.predict(input);
        const values = prediction.dataSync(); // Get tensor values synchronously

        return {
            calories: Math.round(values[0]),
            protein: Math.round(values[1]),
            intensity: Math.min(10, Math.max(1, Math.round(values[2])))
        };
    }

    mapActivityLevel(level) {
        const map = {
            'Sedentary': 1,
            'Lightly Active': 2,
            'Moderately Active': 3,
            'Very Active': 4,
            'Extremely Active': 5
        };
        return map[level] || 1;
    }
}

// ============================================
// TEMPLATE ENGINE (Expert System)
// ============================================

const EXPERT_SYSTEM = {
    getWorkoutPlan: (intensity, bmiCategory) => {
        const workouts = {
            low: [
                "Walking: 30 minutes daily",
                "Light Yoga: 20 minutes for flexibility",
                "Stretching: 10 minutes morning routine",
                "Beginner Bodyweight: Squats (2x10), Wall Pushups (2x10)"
            ],
            medium: [
                "Jogging/Brisk Walk: 45 minutes (3-4x/week)",
                "Cycling: 30 minutes moderate pace",
                "Strength Training: Squats, Pushups, Lunges (3x12)",
                "Pilates: 30 minute session"
            ],
            high: [
                "HIIT Cardio: 20 minutes intense intervals",
                "Weight Training: Full body compound lifts (4x/week)",
                "Running: 5km at steady pace",
                "Sports: Swimming, Basketball, or Tennis (1 hour)"
            ]
        };

        let level = 'low';
        if (intensity > 4) level = 'medium';
        if (intensity > 7) level = 'high';

        // Customize slightly based on BMI
        let advice = workouts[level];
        if (bmiCategory === 'Obese' && level === 'high') {
            // Safety override for high BMI
            advice = workouts['medium'];
            advice.push("Low Impact Cardio (Swimming/Elliptical) to protect joints");
        }

        return advice;
    },

    getNutritionPlan: (calories, protein, bmiCategory, dietPreference) => {
        const fats = Math.round((calories * 0.25) / 9);
        const carbs = Math.round((calories - (protein * 4) - (fats * 9)) / 4);

        const foodSources = {
            'veg': [
                "Protein: Paneer, Lentils (Dal), Chickpeas, Greek Yogurt, Quinoa",
                "Fats: Almonds, Walnuts, Ghee, Olive Oil",
                "Carbs: Brown Rice, Roti, Oats, Sweet Potato"
            ],
            'non-veg': [
                "Protein: Chicken Breast, Eggs, Fish, Lean Mutton",
                "Fats: Fish Oil, Egg Yolk, Avocado, Nuts",
                "Carbs: Rice, Whole Wheat Bread, Potatoes"
            ],
            'vegan': [
                "Protein: Tofu, Soy Chunks, Lentils, Black Beans, Nutritional Yeast",
                "Fats: Avocado, Flax Seeds, Chia Seeds, Coconut Oil",
                "Carbs: Quinoa, Buckwheat, Fruits, Vegetables"
            ]
        };

        const selectedFoods = foodSources[dietPreference] || foodSources['veg'];

        return {
            macros: { protein, fats, carbs },
            tips: [
                `Targets: ${protein}g Protein | ${fats}g Fats | ${carbs}g Carbs`,
                ...selectedFoods,
                "Hydration: Drink at least 3-4 liters of water daily.",
                bmiCategory === 'Overweight' || bmiCategory === 'Obese' ?
                    "Focus on calorie deficit. Reduce processed sugars." :
                    "Maintain a balanced diet with whole foods.",
            ]
        };
    }
};

// ============================================
// DOM ELEMENTS
// ============================================
const elements = {
    form: document.getElementById('fitnessForm'),
    submitBtn: document.getElementById('submitBtn'),
    btnText: document.querySelector('.btn-text'),
    btnLoader: document.querySelector('.btn-loader'),
    results: document.getElementById('results'),
    resultsContent: document.getElementById('resultsContent'),
    error: document.getElementById('error'),
    errorMessage: document.getElementById('errorMessage'),
    weight: document.getElementById('weight'),
    height: document.getElementById('height'),
    activityLevel: document.getElementById('activityLevel'),
    dietPreference: document.getElementById('dietPreference')
};

// ============================================
// INITIALIZATION
// ============================================
const fitnessAI = new FitnessAI();

document.addEventListener('DOMContentLoaded', () => {
    console.log('FitBuddy Local AI initialized 🧠');
    elements.form.addEventListener('submit', handleFormSubmit);

    // Warmup the library
    console.log("TensorFlow.js version:", tf.version.tfjs);
});

// ============================================
// FORM HANDLING
// ============================================

async function handleFormSubmit(e) {
    e.preventDefault();

    hideError();
    hideResults();

    const formData = getFormData();
    if (!validateFormData(formData)) {
        showError('Please fill in all fields with valid values.');
        return;
    }

    const bmi = calculateBMI(formData.weight, formData.height);
    const bmiCategory = getBMICategory(bmi);

    setLoadingState(true, "Initializing AI Model...");

    try {
        // 1. Train the model (Simulation of "Personalizing")
        await fitnessAI.train((progress, loss) => {
            if (progress % 10 === 0) {
                updateLoadingText(`Training Personal Model... ${progress}%`); // Update button text
            }
        });

        // 2. Predict
        updateLoadingText("Generating Plan...");
        const prediction = fitnessAI.predict(formData.weight, formData.height, formData.activityLevel);

        // 3. Generate Content (Expert System)
        const workoutPlan = EXPERT_SYSTEM.getWorkoutPlan(prediction.intensity, bmiCategory);
        const nutritionPlan = EXPERT_SYSTEM.getNutritionPlan(prediction.calories, prediction.protein, bmiCategory, formData.dietPreference);

        // 4. Display
        const recommendations = formatAIResponse(prediction, workoutPlan, nutritionPlan, bmi, bmiCategory);
        displayResults(recommendations);

    } catch (error) {
        console.error('AI Error:', error);
        showError("Failed to generate plan. Please try again.");
    } finally {
        setLoadingState(false);
    }
}

function getFormData() {
    return {
        weight: parseFloat(elements.weight.value),
        height: parseFloat(elements.height.value),
        activityLevel: elements.activityLevel.value,
        dietPreference: elements.dietPreference.value
    };
}

function validateFormData(data) {
    return data.weight > 0 && data.height > 0 && data.activityLevel !== '' && data.dietPreference !== '';
}

function calculateBMI(weight, height) {
    const heightInMeters = height / 100;
    return (weight / (heightInMeters * heightInMeters)).toFixed(1);
}

function getBMICategory(bmi) {
    if (bmi < 18.5) return 'Underweight';
    if (bmi < 25) return 'Normal weight';
    if (bmi < 30) return 'Overweight';
    return 'Obese';
}

// ============================================
// UI FUNCTIONS
// ============================================

function setLoadingState(isLoading, text = "Loading...") {
    if (isLoading) {
        elements.submitBtn.disabled = true;
        elements.btnText.textContent = text;
        elements.btnLoader.style.display = 'inline-block';
    } else {
        elements.submitBtn.disabled = false;
        elements.btnText.textContent = "Get Recommendations";
        elements.btnLoader.style.display = 'none';
    }
}

function updateLoadingText(text) {
    elements.btnText.textContent = text;
}

function displayResults(htmlContent) {
    elements.resultsContent.innerHTML = htmlContent;
    elements.results.style.display = 'block';
    setTimeout(() => {
        elements.results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

function hideResults() {
    elements.results.style.display = 'none';
}

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.error.style.display = 'block';
    setTimeout(() => {
        elements.error.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

function hideError() {
    elements.error.style.display = 'none';
}

function formatAIResponse(prediction, workoutPlan, nutritionPlan, bmi, bmiCategory) {
    return `
        <h3>📊 Personal Analysis</h3>
        <p><strong>BMI:</strong> ${bmi} (${bmiCategory})</p>
        <p><strong>Estimated Daily Needs:</strong> ${prediction.calories} kcal</p>
        
        <h3>💪 Training Plan (Intensity: ${prediction.intensity}/10)</h3>
        <ul>
            ${workoutPlan.map(item => `<li>${item}</li>`).join('')}
        </ul>

        <h3>🥗 Nutrition Strategy</h3>
        <p><strong>Target Macros:</strong> Protein: ${nutritionPlan.macros.protein}g | Carbs: ${nutritionPlan.macros.carbs}g | Fats: ${nutritionPlan.macros.fats}g</p>
        <ul>
            ${nutritionPlan.tips.map(tip => `<li>${tip}</li>`).join('')}
        </ul>
        
        <p><em>Generated by Local Client-Side AI (TensorFlow.js) 🧠</em></p>
    `;
}
