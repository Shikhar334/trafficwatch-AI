import React from 'react';
import { Card } from '@/components/ui/card';
import { Video, AlertTriangle, Gauge, Navigation } from 'lucide-react';

const StatsPanel = ({ stats }) => {
  const statCards = [
    {
      title: 'Total Videos',
      value: stats.total_videos,
      icon: Video,
      color: 'blue',
      testId: 'stat-total-videos'
    },
    {
      title: 'Total Violations',
      value: stats.total_violations,
      icon: AlertTriangle,
      color: 'red',
      testId: 'stat-total-violations'
    },
    {
      title: 'No Helmet',
      value: stats.violations_by_type?.no_helmet || 0,
      icon: AlertTriangle,
      color: 'orange',
      testId: 'stat-no-helmet'
    },
    {
      title: 'Overspeeding',
      value: stats.violations_by_type?.overspeeding || 0,
      icon: Gauge,
      color: 'yellow',
      testId: 'stat-overspeeding'
    }
  ];

  const getColorClasses = (color) => {
    const colors = {
      blue: 'bg-blue-600/20 text-blue-400',
      red: 'bg-red-600/20 text-red-400',
      orange: 'bg-orange-600/20 text-orange-400',
      yellow: 'bg-yellow-600/20 text-yellow-400'
    };
    return colors[color] || colors.blue;
  };

  return (
    <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
      {statCards.map((stat) => {
        const Icon = stat.icon;
        return (
          <Card
            key={stat.title}
            data-testid={stat.testId}
            className="bg-slate-900 border-slate-800 p-6 hover:border-slate-700 transition-colors"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm">{stat.title}</p>
                <p className="text-3xl font-bold text-white mt-2">{stat.value}</p>
              </div>
              <div className={`rounded-xl p-3 ${getColorClasses(stat.color)}`}>
                <Icon className="w-6 h-6" />
              </div>
            </div>
          </Card>
        );
      })}
    </div>
  );
};

export default StatsPanel;
