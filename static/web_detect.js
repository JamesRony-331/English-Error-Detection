const $ = selector => document.querySelector(selector);
const input = $('#image'), button = $('#submit'), status = $('#status');
const preview = $('#preview'), text = $('#text'), result = $('#result');
const errors = $('#errors'), model = $('#model'), video = $('#camera-video');
const openCamera = $('#open-camera'), capture = $('#capture'), closeCamera = $('#close-camera');
let url = null, stream = null, busy = false, opening = false, cameraRequest = 0;

function controls() {
  button.disabled = input.disabled = busy;
  openCamera.disabled = busy || opening || !!stream;
  capture.disabled = busy || !stream || video.readyState < 2;
  closeCamera.disabled = !stream && !opening;
}
function stopCamera() {
  cameraRequest++;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  opening = false;
  video.srcObject = null;
  video.hidden = true;
  controls();
}
function clearResults() {
  text.value = ''; result.textContent = ''; errors.replaceChildren(); model.textContent = '';
}
function showImage(file) {
  if (url) URL.revokeObjectURL(url);
  url = file ? URL.createObjectURL(file) : null;
  preview.hidden = !file; $('#placeholder').hidden = !!file;
  if (url) preview.src = url;
}
input.addEventListener('change', () => {
  stopCamera(); clearResults(); showImage(input.files[0]);
  status.textContent = input.files[0] ? '已选择图片，点击“识别并检测”。' : '请选择图片。';
});
openCamera.addEventListener('click', async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    status.textContent = '浏览器不支持摄像头访问，请使用 localhost / 127.0.0.1 或 HTTPS，或拍照上传。';
    return;
  }
  const requestId = ++cameraRequest;
  opening = true; controls(); status.textContent = '正在请求摄像头权限，请在浏览器中允许访问。';
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({audio: false, video: {
      facingMode: {ideal: 'environment'}, width: {ideal: 1920}, height: {ideal: 1080}
    }});
    // 用户在等待权限时关闭了相机，不保留迟到的视频流。
    if (requestId !== cameraRequest) { acquired.getTracks().forEach(t => t.stop()); return; }
    stream = acquired; video.srcObject = stream; video.hidden = false;
    stream.getVideoTracks().forEach(track => track.addEventListener('ended', () => {
      stopCamera(); status.textContent = '摄像头已断开，可重新打开或上传图片。';
    }));
    await video.play();
    if (requestId === cameraRequest) status.textContent = '请正对作文并保持清晰，点击“拍照并识别”。';
  } catch (error) {
    if (requestId !== cameraRequest) return;
    stopCamera();
    const messages = {NotAllowedError: '摄像头权限被拒绝，请在浏览器设置中允许访问，或拍照上传。',
      NotFoundError: '未找到摄像头，请连接摄像头或上传图片。',
      NotReadableError: '摄像头可能被其他应用占用，请关闭占用应用后重试。'};
    status.textContent = messages[error.name] || '无法打开摄像头，请尝试拍照上传。';
  } finally {
    if (requestId === cameraRequest) opening = false;
    controls();
  }
});
video.addEventListener('loadeddata', controls);
closeCamera.addEventListener('click', () => { stopCamera(); status.textContent = '摄像头已关闭。'; });
window.addEventListener('pagehide', () => { stopCamera(); if (url) URL.revokeObjectURL(url); });

async function recognize(file) {
  if (!file || busy) return;
  if (file.size > 12 * 1024 * 1024) { status.textContent = '图片不能超过 12 MB。'; return; }
  stopCamera(); busy = true; controls(); clearResults(); showImage(file);
  status.textContent = '正在识别图片并加载检测模型，请稍候…';
  try {
    const form = new FormData(); form.append('image', file);
    const response = await fetch('/api/recognize', {method: 'POST', body: form});
    const data = await response.json();
    if (!response.ok) throw Error(data.error || '请求失败');
    preview.src = data.image; text.value = data.text;
    let cursor = 0;
    for (const token of data.tokens) {
      result.append(document.createTextNode(data.text.slice(cursor, token.start)));
      const span = document.createElement(token.error ? 'mark' : 'span');
      span.textContent = data.text.slice(token.start, token.end);
      span.title = `错误概率 ${(token.probability * 100).toFixed(1)}%${token.unknown ? ' · 未登录词' : ''}`;
      result.append(span); cursor = token.end;
      if (token.error) {
        const row = document.createElement('tr');
        for (const value of [span.textContent, `${(token.probability * 100).toFixed(1)}%`, token.unknown ? '未登录词' : '已收录']) {
          const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
        }
        errors.append(row);
      }
    }
    result.append(document.createTextNode(data.text.slice(cursor)));
    status.textContent = data.detection_error || data.message || `识别完成，检测到 ${data.tokens.filter(t => t.error).length} 个疑似错误词。`;
    if (data.model) model.textContent = '模型：' + data.model;
  } catch (error) { status.textContent = error.message; }
  finally { busy = false; controls(); }
}
capture.addEventListener('click', async () => {
  if (!stream || busy || !video.videoWidth || !video.videoHeight) {
    status.textContent = '摄像头画面尚未就绪，请稍候。'; return;
  }
  busy = true; controls();
  try {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .95));
    if (!blob) throw Error('拍照失败，请重试或上传图片。');
    stopCamera(); input.value = ''; busy = false;
    await recognize(new File([blob], 'camera.jpg', {type: 'image/jpeg'}));
  } catch (error) { status.textContent = error.message; }
  finally { busy = false; controls(); }
});
$('#form').addEventListener('submit', event => { event.preventDefault(); recognize(input.files[0]); });
