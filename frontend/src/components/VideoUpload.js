import React, { useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload, Loader2 } from 'lucide-react';

const VideoUpload = ({ onUpload, loading }) => {
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      onUpload(file);
    }
  };

  return (
    <Card className="bg-slate-900 border-slate-800 p-6">
      <h3 className="text-lg font-bold text-white mb-4">Upload Video</h3>
      <div className="space-y-4">
        <div
          data-testid="upload-dropzone"
          onClick={() => !loading && fileInputRef.current?.click()}
          className="border-2 border-dashed border-slate-700 rounded-xl p-8 text-center cursor-pointer hover:border-blue-500 hover:bg-slate-800/50 transition-all"
        >
          {loading ? (
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
              <p className="text-slate-400">Uploading...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <div className="bg-blue-600/20 rounded-full p-4">
                <Upload className="w-8 h-8 text-blue-500" />
              </div>
              <div>
                <p className="text-white font-medium">Click to upload video</p>
                <p className="text-xs text-slate-400 mt-1">MP4, AVI, MOV supported</p>
              </div>
            </div>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="hidden"
          data-testid="video-file-input"
        />
      </div>
    </Card>
  );
};

export default VideoUpload;
