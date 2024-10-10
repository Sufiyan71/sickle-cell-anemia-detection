import React, { useState } from 'react';
import './image.css';

const image = () => {
  const [fileName, setFileName] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setFileName(file ? file.name : null);
  };

  return (
    <div className="upload-section">
      <div className="file-drop-area">
        <label htmlFor="file-upload" className="file-upload-label">
          <img
            src="https://cdn-icons-png.flaticon.com/512/1375/1375450.png" 
            alt="Upload Icon"
            className="upload-icon"
          />
          <p>{fileName ? fileName : "Drag and drop a file here or click to upload"}</p>
        </label>
        <input
          id="file-upload"
          type="file"
          className="file-input"
          onChange={handleFileChange}
        />
      </div>
      <button className="submit-btn">Submit</button>
    </div>
  );
};

export default image;
