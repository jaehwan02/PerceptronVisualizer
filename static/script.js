window.addEventListener('load', () => {
	const canvas = document.getElementById('canvas-box');
	const ctx = canvas.getContext('2d');
	// 초기 배경 흰색
	ctx.fillStyle = 'white';
	ctx.fillRect(0, 0, canvas.width, canvas.height);

	let drawing = false;
	let lastMoveTime = 0;
	const debounceTime = 300; // ms

	// 마우스 드래그
	canvas.addEventListener('mousedown', () => (drawing = true));
	canvas.addEventListener('mouseup', () => (drawing = false));
	canvas.addEventListener('mouseleave', () => (drawing = false));

	canvas.addEventListener('mousemove', (e) => {
		if (!drawing) return;
		ctx.fillStyle = 'black';
		ctx.beginPath();
		ctx.arc(e.offsetX, e.offsetY, 8, 0, 2 * Math.PI);
		ctx.fill();

		const now = Date.now();
		if (now - lastMoveTime > debounceTime) {
			lastMoveTime = now;
			sendImageToServer();
		}
	});

	// Clear 버튼
	document.getElementById('clear-btn').addEventListener('click', () => {
		ctx.fillStyle = 'white';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		document.getElementById('prediction-result').textContent = '예측 결과: -';
		document.getElementById('result-img').src = '';
	});

	async function sendImageToServer() {
		// 1) 캔버스 픽셀
		const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
		const data = imageData.data; // [r,g,b,a, r,g,b,a, ...]

		// 2) bounding box
		let minX = Infinity,
			maxX = -1,
			minY = Infinity,
			maxY = -1;
		for (let y = 0; y < 200; y++) {
			for (let x = 0; x < 200; x++) {
				const idx = (y * 200 + x) * 4;
				const r = data[idx],
					g = data[idx + 1],
					b = data[idx + 2];
				const gray = (r + g + b) / 3;
				if (gray < 220) {
					if (x < minX) minX = x;
					if (x > maxX) maxX = x;
					if (y < minY) minY = y;
					if (y > maxY) maxY = y;
				}
			}
		}

		if (minX > maxX || minY > maxY) {
			minX = 90;
			maxX = 110;
			minY = 90;
			maxY = 110;
		}

		const bbW = maxX - minX + 1;
		const bbH = maxY - minY + 1;

		// 3) 임시 캔버스 tmpCanvas에 중앙 배치
		const tmpCanvas = document.createElement('canvas');
		tmpCanvas.width = 200;
		tmpCanvas.height = 200;
		const tmpCtx = tmpCanvas.getContext('2d');
		tmpCtx.fillStyle = 'white';
		tmpCtx.fillRect(0, 0, 200, 200);

		const centerX = Math.floor((200 - bbW) / 2);
		const centerY = Math.floor((200 - bbH) / 2);

		tmpCtx.drawImage(canvas, minX, minY, bbW, bbH, centerX, centerY, bbW, bbH);

		// 4) tmpCanvas -> 8x8 다운샘플
		const blockSize = 25; // 200/8
		const tmpData = tmpCtx.getImageData(0, 0, 200, 200).data;
		const downsampled = [];

		for (let row = 0; row < 8; row++) {
			for (let col = 0; col < 8; col++) {
				let sumGray = 0,
					count = 0;
				for (let y = 0; y < blockSize; y++) {
					for (let x = 0; x < blockSize; x++) {
						const px = (row * blockSize + y) * 200 + (col * blockSize + x);
						const idx = px * 4;
						const r = tmpData[idx],
							g = tmpData[idx + 1],
							b = tmpData[idx + 2];
						const gray = (r + g + b) / 3;
						sumGray += gray;
						count++;
					}
				}
				let avg = sumGray / count; // 0..255
				// 학습 시 X/=16.0 -> 여기서는 (avg/255)*16
				avg = (avg / 255) * 16.0;
				downsampled.push(avg);
			}
		}
		for (let i = 0; i < downsampled.length; i++) {
			downsampled[i] = 16 - downsampled[i]; // 값 반전
		}
		// 5) 서버 전송
		const response = await fetch('/predict', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ pixels: downsampled }),
		});
		if (!response.ok) return;

		const result = await response.json();
		document.getElementById('prediction-result').textContent = `예측 결과: ${result.prediction}`;
		document.getElementById('result-img').src = 'data:image/png;base64,' + result.img_data;
	}
});
