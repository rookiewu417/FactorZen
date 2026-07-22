import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import type { TrackPoint } from '../types'

interface IcChartProps {
  points: TrackPoint[]
  height?: number
}

/** 向前追踪 IC 折线图 */
export function IcChart({ points, height = 280 }: IcChartProps) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current || points.length === 0) return

    const chart = echarts.init(ref.current)
    chart.setOption({
      tooltip: { trigger: 'axis' },
      grid: { left: 48, right: 24, top: 32, bottom: 40 },
      xAxis: {
        type: 'category',
        data: points.map((p) => p.date ?? ''),
        boundaryGap: false,
      },
      yAxis: {
        type: 'value',
        scale: true,
        name: 'IC',
      },
      series: [
        {
          name: 'IC',
          type: 'line',
          data: points.map((p) => p.ic),
          showSymbol: points.length < 40,
          lineStyle: { width: 2 },
          areaStyle: { opacity: 0.06 },
        },
      ],
    })

    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      chart.dispose()
    }
  }, [points])

  return <div ref={ref} style={{ width: '100%', height }} />
}
