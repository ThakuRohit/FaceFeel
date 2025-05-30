const video = document.querySelector("#videoElement");

navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
  })
  .catch((err) => {
    console.error("Error accessing camera: ", err);
  });

function captureAndSend() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg");

  fetch("http://127.0.0.1:8000/process-image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl })
  })
    .then(response => response.json())
    .then(data => {
      document.getElementById("emojiDisplay").innerHTML =
        `<img src="${data.emoji}" style="width:200px; border-radius:10px;">`;
      document.getElementById("emotionText").textContent =
        `Emotion: ${data.emotion}`;

      // Show download button
      const link = document.getElementById("downloadEmoji");
      link.href = data.emoji;
      link.style.display = "inline-block";
    })
    .catch(error => {
      console.error("Error:", error);
      alert("Failed to get emoji. Is your backend running?");
    });
}
