import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

interface NavChartProps {
  /** NAV 序列: [date, nav] */
  data: [string, number][]
  height?: number
}

/** 轻量 ECharts 折线图封装 */
export function NavChart({ data, height = 360 }: NavChartProps) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current || data.length === 0) return

    const chart = echarts.init(ref.current)
    chart.setOption({
      tooltip: { trigger: 'axis' },
      grid: { left: 56, right: 24, top: 32, bottom: 40 },
      xAxis: {
        type: 'category',
        data: data.map(([d]) => d),
        boundaryGap: false,
      },
      yAxis: {
        type: 'value',
        scale: true,
      },
      series: [
        {
          name: 'NAV',
          type: 'line',
          data: data.map(([, v]) => v),
          showSymbol: data.length < 80,
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
  }, [data])

  return <div ref={ref} style={{ width: '100%', height }} />
}
