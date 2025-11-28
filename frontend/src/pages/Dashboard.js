import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';
import { Upload, Video, AlertTriangle, Activity, LogOut, Gauge, Play, Download } from 'lucide-react';
import VideoUpload from '../components/VideoUpload';
import ViolationsList from '../components/ViolationsList';
import CalibrationPanel from '../components/CalibrationPanel';
import StatsPanel from '../components/StatsPanel';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = ({ user, setUser }) => {
  const [videos, setVideos] = useState([]);
  const [violations, setViolations] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('videos');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [videosRes, violationsRes, statsRes] = await Promise.all([
        axios.get(`${API}/videos`, { withCredentials: true }),
        axios.get(`${API}/violations`, { withCredentials: true }),
        axios.get(`${API}/stats`, { withCredentials: true })
      ]);
      setVideos(videosRes.data);
      setViolations(violationsRes.data);
      setStats(statsRes.data);
    } catch (error) {
      console.error('Error loading data:', error);
    }
  };

  const handleLogout = async () => {
    try {
      await axios.post(`${API}/auth/logout`, null, { withCredentials: true });
      setUser(null);
      window.location.href = '/';
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const handleVideoUpload = async (file) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API}/videos/upload`, formData, {
        withCredentials: true,
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      toast.success('Video uploaded successfully!');
      
      // Start processing
      await axios.post(`${API}/videos/${response.data.id}/process`, null, {
        withCredentials: true
      });
      
      toast.info('Processing started...');
      loadData();
    } catch (error) {
      toast.error('Upload failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (videoId) => {
    try {
      window.open(`${API}/videos/${videoId}/download`, '_blank');
    } catch (error) {
      toast.error('Download failed');
    }
  };

  return (
    <div className="min-h-screen bg-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-blue-600 rounded-xl p-2">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">TrafficWatch AI</h1>
                <p className="text-xs text-slate-400">Traffic Violation Detection</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right hidden sm:block">
                <p className="text-sm font-medium text-white">{user.name}</p>
                <p className="text-xs text-slate-400">{user.email}</p>
              </div>
              <Button
                data-testid="logout-button"
                onClick={handleLogout}
                variant="outline"
                size="sm"
                className="border-slate-700 text-slate-300 hover:bg-slate-800"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-8">
        {/* Stats Overview */}
        {stats && <StatsPanel stats={stats} />}

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="mt-8">
          <TabsList className="bg-slate-900 border border-slate-800">
            <TabsTrigger value="videos" data-testid="videos-tab" className="data-[state=active]:bg-blue-600">
              <Video className="w-4 h-4 mr-2" />
              Videos
            </TabsTrigger>
            <TabsTrigger value="violations" data-testid="violations-tab" className="data-[state=active]:bg-blue-600">
              <AlertTriangle className="w-4 h-4 mr-2" />
              Violations
            </TabsTrigger>
            <TabsTrigger value="calibration" data-testid="calibration-tab" className="data-[state=active]:bg-blue-600">
              <Gauge className="w-4 h-4 mr-2" />
              Calibration
            </TabsTrigger>
          </TabsList>

          {/* Videos Tab */}
          <TabsContent value="videos" className="mt-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1">
                <VideoUpload onUpload={handleVideoUpload} loading={loading} />
              </div>
              <div className="lg:col-span-2">
                <Card className="bg-slate-900 border-slate-800 p-6">
                  <h3 className="text-lg font-bold text-white mb-4">Uploaded Videos</h3>
                  <div className="space-y-3">
                    {videos.length === 0 ? (
                      <p className="text-slate-400 text-center py-8">No videos uploaded yet</p>
                    ) : (
                      videos.map((video) => (
                        <div
                          key={video.id}
                          data-testid={`video-item-${video.id}`}
                          className="bg-slate-800 rounded-lg p-4 flex items-center justify-between hover:bg-slate-750 transition-colors"
                        >
                          <div className="flex items-center gap-3">
                            <div className="bg-blue-600/20 rounded-lg p-2">
                              <Play className="w-5 h-5 text-blue-400" />
                            </div>
                            <div>
                              <p className="text-white font-medium text-sm">{video.filename}</p>
                              <div className="flex items-center gap-3 mt-1">
                                <span className={`text-xs px-2 py-1 rounded ${
                                  video.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                                  video.status === 'processing' ? 'bg-yellow-500/20 text-yellow-400' :
                                  video.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                                  'bg-slate-700 text-slate-400'
                                }`}>
                                  {video.status}
                                </span>
                                <span className="text-xs text-slate-400">{video.duration.toFixed(1)}s</span>
                                {video.total_violations > 0 && (
                                  <span className="text-xs text-red-400">{video.total_violations} violations</span>
                                )}
                              </div>
                            </div>
                          </div>
                          {video.status === 'completed' && (
                            <Button
                              data-testid={`download-video-${video.id}`}
                              size="sm"
                              onClick={() => handleDownload(video.id)}
                              className="bg-blue-600 hover:bg-blue-700"
                            >
                              <Download className="w-4 h-4" />
                            </Button>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Violations Tab */}
          <TabsContent value="violations" className="mt-6">
            <ViolationsList violations={violations} videos={videos} />
          </TabsContent>

          {/* Calibration Tab */}
          <TabsContent value="calibration" className="mt-6">
            <CalibrationPanel onCalibrate={loadData} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Dashboard;
