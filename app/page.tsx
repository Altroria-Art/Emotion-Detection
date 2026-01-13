"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<string>("ยังไม่เริ่ม");
  const [emotion, setEmotion] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  async function loadOpenCV() {
    if (typeof window === "undefined") return;

    // ready แล้ว
    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }

    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;

      script.onload = () => {
        const cv = (window as any).cv;
        if (!cv)
          return reject(new Error("OpenCV โหลดแล้วแต่ window.cv ไม่มีค่า"));

        const waitReady = () => {
          if ((window as any).cv?.Mat) {
            cvRef.current = (window as any).cv;
            resolve();
          } else {
            setTimeout(waitReady, 50);
          }
        };

        // บาง build มี callback บาง build พร้อมทันที
        if ("onRuntimeInitialized" in cv) {
          cv.onRuntimeInitialized = () => waitReady();
        } else {
          waitReady();
        }
      };

      script.onerror = () =>
        reject(new Error("โหลด /opencv/opencv.js ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }

  // Load Haar cascade file into OpenCV FS
  async function loadCascade() {
    const cv = cvRef.current;
    if (!cv) throw new Error("cv ยังไม่พร้อม");

    const cascadeUrl = "/opencv/haarcascade_frontalface_default.xml";
    const res = await fetch(cascadeUrl);
    if (!res.ok) throw new Error("โหลด cascade ไม่สำเร็จ");
    const data = new Uint8Array(await res.arrayBuffer());

    // เขียนไฟล์ลง OpenCV virtual FS
    const cascadePath = "haarcascade_frontalface_default.xml";
    try {
      cv.FS_unlink(cascadePath);
    } catch {}
    cv.FS_createDataFile("/", cascadePath, data, true, false, false);

    const faceCascade = new cv.CascadeClassifier();
    const loaded = faceCascade.load(cascadePath);
    if (!loaded) throw new Error("cascade load() ไม่สำเร็จ");
    faceCascadeRef.current = faceCascade;
  }

  // 3) Load ONNX model + classes
  async function loadModel() {
    const session = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      { executionProviders: ["wasm"] }
    );
    sessionRef.current = session;

    const clsRes = await fetch("/models/classes.json");
    if (!clsRes.ok) throw new Error("โหลด classes.json ไม่สำเร็จ");
    classesRef.current = await clsRes.json();
  }

  // 4) Start camera
  async function startCamera() {
    setStatus("ขอสิทธิ์กล้อง...");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    if (!videoRef.current) return;
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
    setStatus("กำลังทำงาน...");
    requestAnimationFrame(loop);
  }

  // 5) Preprocess face ROI -> tensor
  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    // YOLO classification มักรับ input เป็น [1,3,H,W] float32 (0..1)
    // เพื่อให้ง่าย: resize เป็น 64x64 และทำ RGB
    const size = 128;
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;
    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);

    const imgData = ctx.getImageData(0, 0, size, size).data; // RGBA
    const float = new Float32Array(1 * 3 * size * size);

    // CHW
    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        const r = imgData[i * 4 + 0] / 255;
        const g = imgData[i * 4 + 1] / 255;
        const b = imgData[i * 4 + 2] / 255;
        float[idx++] = c === 0 ? r : c === 1 ? g : b;
      }
    }

    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  // 6) Softmax
  function softmax(logits: Float32Array) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  // 7) Main loop
  async function loop() {
    try {
      const cv = cvRef.current;
      const faceCascade = faceCascadeRef.current;
      const session = sessionRef.current;
      const classes = classesRef.current;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!cv || !faceCascade || !session || !classes || !video || !canvas) {
        requestAnimationFrame(loop);
        return;
      }

      const ctx = canvas.getContext("2d")!;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      // OpenCV: read frame
      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      const faces = new cv.RectVector();
      const msize = new cv.Size(0, 0);
      faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

      // วาดกรอบ + เลือกใบหน้าที่ใหญ่สุด
      let bestRect: any = null;
      let bestArea = 0;

      for (let i = 0; i < faces.size(); i++) {
        const r = faces.get(i);
        const area = r.width * r.height;
        if (area > bestArea) {
          bestArea = area;
          bestRect = r;
        }
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 2;
        ctx.strokeRect(r.x, r.y, r.width, r.height);
      }

      if (bestRect) {
        // crop face into a small canvas
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = bestRect.width;
        faceCanvas.height = bestRect.height;
        const fctx = faceCanvas.getContext("2d")!;
        fctx.drawImage(
          canvas,
          bestRect.x,
          bestRect.y,
          bestRect.width,
          bestRect.height,
          0,
          0,
          bestRect.width,
          bestRect.height
        );

        // run onnx
        const input = preprocessToTensor(faceCanvas);

        // ชื่อ input/output อาจต่างกันตามการ export
        // วิธีง่าย: ใช้ key ตัวแรกของ session.inputNames
        const feeds: Record<string, ort.Tensor> = {};
        feeds[session.inputNames[0]] = input;

        const out = await session.run(feeds);
        const outName = session.outputNames[0];
        const logits = out[outName].data as Float32Array;

        const probs = softmax(logits);
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        setEmotion(classes[maxIdx] ?? `class_${maxIdx}`);
        setConf(probs[maxIdx] ?? 0);

        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(bestRect.x, Math.max(0, bestRect.y - 28), 220, 28);
        ctx.fillStyle = "white";
        ctx.font = "16px sans-serif";
        ctx.fillText(
          `${classes[maxIdx]} ${(probs[maxIdx] * 100).toFixed(1)}%`,
          bestRect.x + 6,
          bestRect.y - 8
        );
      }

      // cleanup
      src.delete();
      gray.delete();
      faces.delete();

      requestAnimationFrame(loop);
    } catch (e: any) {
      setStatus(`ผิดพลาด: ${e?.message ?? e}`);
    }
  }

  // Boot sequence
  useEffect(() => {
    (async () => {
      try {
        setStatus("กำลังโหลด OpenCV...");
        await loadOpenCV();

        setStatus("กำลังโหลด Haar cascade...");
        await loadCascade();

        setStatus("กำลังโหลดโมเดล ONNX...");
        await loadModel();

        setStatus("พร้อม เริ่มกดปุ่ม Start");
      } catch (e: any) {
        setStatus(`เริ่มต้นไม่สำเร็จ: ${e?.message ?? e}`);
      }
    })();
  }, []);

  return (
    <main className="min-h-screen p-10 flex flex-col gap-6 items-start">
      {/* กล่องหัวข้อและสถานะ */}
      <div className="bg-[#E9C48C]/90 backdrop-blur-sm p-6 rounded-[20px] shadow-xl border border-white/20 min-w-[300px]">
        <h1 className="text-2xl font-bold text-gray-800 mb-4">Face Emotion</h1>
        <div className="space-y-2 text-gray-700">
          <p className="text-lg">สถานะ: {status}</p>
          <p className="text-xl font-bold">
            Emotion: {emotion} |{" "}
            <span className="text-sm font-normal text-gray-500">
              Conf: {(conf * 100).toFixed(1)}%
            </span>
          </p>
        </div>
      </div>

      {/* ปุ่มกด */}
      <button
        className="px-8 py-3 rounded-xl bg-[#C1E1C1] hover:bg-[#A8D1A8] text-gray-800 font-bold shadow-lg transition-transform active:scale-95 flex items-center gap-2"
        onClick={startCamera}
      >
        Start Camera 📷
      </button>

      {/* พื้นที่แสดงกล้อง */}
      <div className="relative rounded-lg overflow-hidden border-4 border-[#3A7EBF] shadow-2xl bg-black max-w-2xl w-full">
        {/* แถบชื่อหน้าต่างจำลอง (ถ้าอยากได้สไตล์ Windows) */}
        <div className="bg-[#3A7EBF] text-white px-3 py-1 text-sm flex justify-between items-center">
          <span>Camera</span>
          <div className="flex gap-1">
            <div className="w-3 h-3 bg-white/30 rounded-sm"></div>
            <div className="w-3 h-3 bg-white/30 rounded-sm"></div>
            <div className="w-3 h-3 bg-red-500 rounded-sm"></div>
          </div>
        </div>

        <video ref={videoRef} className="hidden" playsInline />
        <canvas ref={canvasRef} className="w-full block" />
      </div>

      <p className="text-sm font-bold text-gray-800 bg-white/40 px-2 py-1 rounded">
        หมายเหตุ: ต้องกดปุ่ม Start เพื่อขอสิทธิ์เปิดกล้อง
      </p>
    </main>
  );
}
