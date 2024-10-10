// components/Image/image.jsx
import React, { useState } from 'react';

const Image = () => {
    const [file, setFile] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [probabilities, setProbabilities] = useState({});
    const [imagePath, setImagePath] = useState('');

    const handleChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            if (response.ok) {
                setPrediction(data.prediction);
                setProbabilities(data.probabilities);
                setImagePath(data.image_path);
            } else {
                console.error(data.error);
            }
        } catch (error) {
            console.error('Error:', error);
        }
    };

    return (
        <div>
            <h1>Sickle Cell Detection</h1>
            <form onSubmit={handleSubmit}>
                <input type="file" accept="image/*" onChange={handleChange} required />
                <button type="submit">Upload and Predict</button>
            </form>

            {prediction && (
                <div>
                    <h2>Prediction: {prediction}</h2>
                    <h3>Probabilities:</h3>
                    <ul>
                        {Object.entries(probabilities).map(([className, prob]) => (
                            <li key={className}>
                                {className}: {prob}%
                            </li>
                        ))}
                    </ul>
                    <img src={imagePath} alt="Uploaded" style={{ maxWidth: '300px' }} />
                </div>
            )}
        </div>
    );
};

export default Image;
