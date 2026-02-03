import React, { useState, useEffect } from 'react';
import { Camera, Upload, Leaf, Info, AlertTriangle, CheckCircle, Loader } from 'lucide-react';
import { preprocessImage, runInference } from './utils';

export default function App() {
    const [image, setImage] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [diseaseInfo, setDiseaseInfo] = useState({});
    const [errorHeader, setErrorHeader] = useState("");

    useEffect(() => {
        fetch('/disease_info.json')
            .then(res => res.json())
            .then(data => setDiseaseInfo(data))
            .catch(err => console.error("Failed to load disease info:", err));
    }, []);

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImage(file);
            setPreviewUrl(URL.createObjectURL(file));
            setResult(null);
            setErrorHeader("");
        }
    };

    const analyzeImage = async () => {
        if (!image) return;
        setLoading(true);
        setResult(null);
        setErrorHeader("");

        try {
            const imgElement = new Image();
            imgElement.src = previewUrl;
            await imgElement.decode();

            const tensor = await preprocessImage(imgElement);
            const output = await runInference('/model.onnx', tensor);

            setResult(output);
        } catch (err) {
            console.error(err);
            setErrorHeader(err.message || "Failed to analyze image");
        } finally {
            setLoading(false);
        }
    };

    const getDiseaseKey = (label) => {
        const mapping = {
            "Bacterial Spot": "Tomato___Bacterial_spot",
            "Early Blight": "Tomato___Early_blight",
            "Late Blight": "Tomato___Late_blight",
            "Leaf Mold": "Tomato___Leaf_Mold",
            "Septoria Leaf Spot": "Tomato___Septoria_leaf_spot",
            "Spider Mites": "Tomato___Spider_mites Two-spotted_spider_mite",
            "Target Spot": "Tomato___Target_Spot",
            "Yellow Leaf Curl Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Mosaic Virus": "Tomato___Tomato_mosaic_virus",
            "Healthy": "Tomato___healthy"
        };
        return mapping[label];
    };

    const info = result ? diseaseInfo[getDiseaseKey(result.label)] : null;

    return (
        <div className="min-h-screen bg-slate-50 p-4 font-sans text-slate-900">
            <header className="max-w-md mx-auto mb-8 text-center pt-8">
                <div className="flex items-center justify-center gap-2 mb-2">
                    <Leaf className="text-emerald-600 w-8 h-8" />
                    <h1 className="text-3xl font-extrabold tracking-tight text-slate-800">TomatoCare</h1>
                </div>
                <p className="text-slate-500 text-sm">Lightweight Test Version</p>
            </header>

            <main className="max-w-md mx-auto">
                <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/50 p-6 mb-6 border border-slate-100">
                    <div className="relative aspect-square rounded-2xl bg-slate-50 border-2 border-dashed border-slate-200 flex flex-col items-center justify-center mb-6 overflow-hidden">
                        {previewUrl ? (
                            <img src={previewUrl} className="w-full h-full object-cover" alt="Preview" />
                        ) : (
                            <div className="text-center px-4">
                                <Upload className="w-12 h-12 text-slate-300 mx-auto mb-3" />
                                <p className="text-slate-500 font-medium">Capture or Upload Leaf Image</p>
                                <p className="text-slate-400 text-xs mt-1">Make sure the leaf is clearly visible</p>
                            </div>
                        )}
                        <input
                            type="file"
                            accept="image/*"
                            onChange={handleImageChange}
                            className="absolute inset-0 opacity-0 cursor-pointer"
                        />
                    </div>

                    {errorHeader && (
                        <div className="mb-4 p-3 bg-red-50 border border-red-100 text-red-700 text-xs rounded-xl flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                            <p>{errorHeader}</p>
                        </div>
                    )}

                    <button
                        onClick={analyzeImage}
                        disabled={!image || loading}
                        className={`w-full py-4 rounded-2xl font-bold flex items-center justify-center gap-2 transition-all active:scale-95 ${!image
                            ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                            : loading
                                ? 'bg-emerald-100 text-emerald-700'
                                : 'bg-emerald-600 text-white shadow-lg shadow-emerald-200 hover:bg-emerald-700'
                            }`}
                    >
                        {loading ? (
                            <><Loader className="w-5 h-5 animate-spin" /> Analyzing Pattern...</>
                        ) : (
                            <>
                                <Camera className="w-5 h-5" />
                                Scan Leaf Now
                            </>
                        )}
                    </button>
                </div>

                {result && (
                    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/50 p-6 mb-12 border border-slate-100">
                            <div className="flex items-start justify-between mb-4">
                                <div>
                                    <h2 className="text-xs uppercase tracking-wider font-bold text-slate-400 mb-1">Diagnosis Result</h2>
                                    <div className="flex items-center gap-2">
                                        <p className={`text-2xl font-black ${result.label === 'Healthy' ? 'text-emerald-600' : 'text-red-600'}`}>
                                            {result.label}
                                        </p>
                                        {result.label === 'Healthy' ? (
                                            <CheckCircle className="text-emerald-500 w-6 h-6" />
                                        ) : (
                                            <AlertTriangle className="text-red-500 w-6 h-6" />
                                        )}
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs font-bold text-slate-300">Confidence</p>
                                    <p className={`text-lg font-mono font-bold ${result.confidence < 0.8 ? 'text-amber-500' : 'text-slate-600'}`}>
                                        {(result.confidence * 100).toFixed(1)}%
                                    </p>
                                </div>
                            </div>

                            {result.confidence < 0.8 && (
                                <div className="mb-6 p-4 bg-amber-50 border border-amber-100 rounded-2xl flex items-start gap-3">
                                    <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                                    <div className="text-xs text-amber-800 leading-relaxed">
                                        <p className="font-bold mb-1 uppercase tracking-wider text-[10px]">Low Confidence Warning</p>
                                        This image doesn't clearly match a known tomato leaf pattern. The prediction below might be inaccurate. Please ensure you're scanning a <strong>tomato leaf</strong> in good lighting.
                                    </div>
                                </div>
                            )}

                            {info && result.label !== 'Healthy' && (
                                <div className="space-y-4 pt-4 border-t border-slate-50">
                                    <div>
                                        <h3 className="flex items-center gap-1.5 text-xs font-bold text-slate-400 uppercase mb-2">
                                            <Info className="w-3.5 h-3.5" /> Treatment Advice
                                        </h3>
                                        <div className="bg-emerald-50 text-emerald-800 p-4 rounded-xl text-sm leading-relaxed">
                                            <p className="font-semibold mb-1">Organic Option:</p>
                                            <ul className="list-disc ml-4 space-y-1">
                                                {info.treatment.organic.map((t, idx) => <li key={idx}>{t}</li>)}
                                            </ul>
                                        </div>
                                    </div>

                                    <div className="bg-slate-50 p-4 rounded-xl text-xs text-slate-500 leading-normal italic">
                                        <p className="font-bold not-italic mb-1 uppercase text-[10px] tracking-widest text-slate-400">🌴 UAE Specific Notes</p>
                                        {info.uae_notes}
                                    </div>
                                </div>
                            )}

                            {result.label === 'Healthy' && (
                                <div className="bg-emerald-50 text-emerald-800 p-6 rounded-2xl text-center border border-emerald-100">
                                    <p className="font-bold text-lg mb-1">Your plant is doing great!</p>
                                    <p className="text-sm opacity-80 italic">Continue regular watering and monitoring in the UAE climate.</p>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}
