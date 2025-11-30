function uploadImage() {
    const input = document.getElementById('imageUpload');
    const file = input.files[0];
    if (!file) {
      alert("Please select an image.");
      return;
    }
  
    const formData = new FormData();
    formData.append('image', file);
  
    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      if (data.flower) {
        document.getElementById('uploadedImage').src = URL.createObjectURL(file);
        document.getElementById('flowerName').textContent = "Flower: " + data.flower;
        document.getElementById('result').classList.remove('hidden');
      } else {
        alert("Prediction failed.");
      }
    })
    .catch(err => {
      alert("An error occurred: " + err);
    });
  }
  
  function goToHome() {
    document.getElementById('result').classList.add('hidden');
  }
  