# Plant Disease Recognition System

Welcome to the Plant Disease Recognition System! 🌿🔍

Our goal is to assist in the swift identification of plant diseases. Simply upload an image of a plant, and our system will analyze it to detect any signs of disease. Let's work together to safeguard our crops and ensure a healthier harvest!

## How It Works
1. **Upload Image:** Navigate to the **Disease Detection** page and upload an image of a plant you suspect might be diseased.
2. **Analysis:** Our system will use advanced algorithms to examine the image and identify potential diseases.
3. **Results:** View the analysis results and get recommendations for further action.

## Why Choose Us?
- **Precision:** Our system employs cutting-edge machine learning techniques for precise disease detection.
- **Ease of Use:** Designed with a simple and intuitive interface for a seamless user experience.
- **Speed and Efficiency:** Get results in seconds, enabling quick decision-making.

## Project Structure
- `app.py`: Main application file containing the Streamlit app.
- `trained_plant_disease_model.keras`: Pre-trained TensorFlow model for plant disease recognition.
- `home_page.jpeg`: Image displayed on the home page.
- `requirements.txt`: List of dependencies required to run the project.

## Dataset
This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on Kaggle. This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure. A new directory containing 33 test images is created later for prediction purposes.

### Content
1. `train` (70295 images)
2. `test` (33 images)
3. `validation` (17572 images)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yuganshgupta/plant-disease-recognition.git
    ```
2. Navigate to the project directory:
    ```bash
    cd plant-disease-recognition
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Open your web browser and go to `http://localhost:8501`.

## Contributing
We welcome contributions! Please read our Contributing Guidelines for more details.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## About Us
Discover more about our project, team, and objectives on the **About** page.

---

Feel free to reach out if you have any questions or need further assistance. Happy coding! 😊
