import React from 'react';
import { Card } from '@/components/ui/card';
import { AlertTriangle, Gauge, Navigation } from 'lucide-react';

const ViolationsList = ({ violations, videos }) => {
  const getViolationIcon = (type) => {
    switch (type) {
      case 'no_helmet':
        return <AlertTriangle className="w-5 h-5 text-red-500" />;
      case 'overspeeding':
        return <Gauge className="w-5 h-5 text-yellow-500" />;
      case 'wrong_way':
        return <Navigation className="w-5 h-5 text-orange-500" />;
      default:
        return <AlertTriangle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getVideoName = (videoId) => {
    const video = videos.find(v => v.id === videoId);
    return video ? video.filename : 'Unknown';
  };

  return (
    <Card className="bg-slate-900 border-slate-800 p-6">
      <h3 className="text-lg font-bold text-white mb-4">All Violations</h3>
      <div className="space-y-3">
        {violations.length === 0 ? (
          <p className="text-slate-400 text-center py-8">No violations detected yet</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="text-left text-slate-400 text-sm font-medium pb-3">Type</th>
                  <th className="text-left text-slate-400 text-sm font-medium pb-3">Video</th>
                  <th className="text-left text-slate-400 text-sm font-medium pb-3">Track ID</th>
                  <th className="text-left text-slate-400 text-sm font-medium pb-3">Time</th>
                  <th className="text-left text-slate-400 text-sm font-medium pb-3">Speed</th>
                  <th className="text-left text-slate-400 text-sm font-medium pb-3">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {violations.map((violation) => (
                  <tr
                    key={violation.id}
                    data-testid={`violation-item-${violation.id}`}
                    className="border-b border-slate-800 hover:bg-slate-800/50 transition-colors"
                  >
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        {getViolationIcon(violation.violation_type)}
                        <span className="text-white text-sm capitalize">
                          {violation.violation_type.replace('_', ' ')}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 text-slate-300 text-sm">
                      {getVideoName(violation.video_id)}
                    </td>
                    <td className="py-3 text-slate-300 text-sm">#{violation.track_id}</td>
                    <td className="py-3 text-slate-300 text-sm">{violation.timestamp.toFixed(2)}s</td>
                    <td className="py-3 text-slate-300 text-sm">
                      {violation.speed ? `${violation.speed.toFixed(1)} km/h` : '-'}
                    </td>
                    <td className="py-3 text-slate-300 text-sm">
                      {(violation.confidence * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Card>
  );
};

export default ViolationsList;
