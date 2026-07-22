import { useCallback, useEffect, useState } from 'react'
import {
  Button,
  Card,
  Form,
  Input,
  Select,
  Space,
  Table,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { Link } from 'react-router-dom'
import {
  fetchFileContent,
  fetchFiles,
  fetchRuns,
  putFileContent,
  submitJob,
} from '../api/client'
import type { RunSummary } from '../types'

const STRATEGY_NAMES = [
  'trend_timing',
  'momentum_rotation',
  'sleeve',
  'quantile_group',
] as const

/** 从 sleeve_top200_h10.py 提炼的最小可跑骨架（真实 import 路径）。 */
const SCRIPT_TEMPLATE = `"""自定义策略回测脚本骨架。

交易轨预置权重回测：t 日信号 → t+1 开盘执行，含涨跌停/ST/停牌/T+1 约束与成本。
请改写 build_weights() 生成 dict[signal_date] -> DataFrame[ts_code, target_weight]。

用法::

    python workspace/configs/<this_script>.py --start 20240101 --end 20241231
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from factorzen.core.experiment import get_git_sha
from factorzen.daily.evaluation.backtest import (
    BacktestConfig,
    PrecomputedWeightsStrategy,
    run_strategy_backtest,
    trim_backtest_to_first_trade,
)
from factorzen.pipelines.combine_backtest import (
    _metrics_from_result,
    build_cost_model_from_bps,
    load_market_panel,
)

OUT_ROOT = Path("workspace/strategies")


def build_weights(trade_dates: list) -> dict:
    """生成预置目标权重。请替换为你的逻辑。

    返回: dict[signal_date] -> DataFrame 列 [ts_code, target_weight]
    """
    weights: dict = {}
    # TODO: 填写权重生成逻辑（可参考 workspace/configs/sleeve_top200_h10.py）
    _ = trade_dates
    return weights


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="20240101")
    ap.add_argument("--end", default="20241231")
    ap.add_argument("--universe", default="all_a")
    ap.add_argument("--run-id", default=None, dest="run_id")
    ap.add_argument("--cost-bps", type=float, default=None, dest="cost_bps")
    args = ap.parse_args()

    market = load_market_panel(
        start=args.start, end=args.end, universe=args.universe, market="ashare"
    )
    price_df = market["price_df"]
    trade_dates = sorted(
        price_df.select("trade_date").unique()["trade_date"].to_list()
    )

    weights = build_weights(trade_dates)
    if not weights:
        raise SystemExit("build_weights() 返回空权重，请先实现策略逻辑")

    strategy = PrecomputedWeightsStrategy(weights)
    strategy.name = "custom_strategy"
    cfg = BacktestConfig(
        factor_col="factor_clean",
        frequency="daily",
        max_abs_weight=1.0,
        max_participation_rate=1.0,
        strategy_type=strategy.name,
        strategy_params={},
        cost_model="linear",
    )
    # 骨架：价格面板 ts_code×date 作 factor_df 占位
    factor_df = (
        price_df.select(["trade_date", "ts_code"])
        .unique()
        .with_columns(pl.lit(0.0).alias("factor_clean"))
    )
    result = run_strategy_backtest(
        strategy,
        factor_df,
        price_df,
        config=cfg,
        cost_model=build_cost_model_from_bps(args.cost_bps),
        factor_name=strategy.name,
        is_st_by_date=market["is_st_by_date"],
    )
    result = trim_backtest_to_first_trade(result)
    metrics = _metrics_from_result(result)

    rid = args.run_id or "custom_strategy"
    out = OUT_ROOT / rid
    out.mkdir(parents=True, exist_ok=True)
    result.nav.write_parquet(str(out / "nav.parquet"))
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": rid,
                "git_sha": get_git_sha(),
                "strategy": strategy.name,
                "start": args.start,
                "end": args.end,
                "universe": args.universe,
                "exec": "t 日信号 → t+1 开盘执行；max_participation_rate=1.0",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[custom] → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
`

function shortSha(sha: string | null): string {
  if (!sha) return '—'
  return sha.length > 8 ? sha.slice(0, 8) : sha
}

export function StrategyPage() {
  const [builtinForm] = Form.useForm()
  const [builtinSubmitting, setBuiltinSubmitting] = useState(false)

  // 自定义脚本
  const [scripts, setScripts] = useState<string[]>([])
  const [selectedScript, setSelectedScript] = useState<string | null>(null)
  const [editor, setEditor] = useState('')
  const [scriptExtra, setScriptExtra] = useState('')
  const [saving, setSaving] = useState(false)
  const [running, setRunning] = useState(false)
  const [newName, setNewName] = useState('')

  // 产物列表
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [runsLoading, setRunsLoading] = useState(true)

  const loadScripts = useCallback(() => {
    fetchFiles('configs')
      .then((res) => {
        const py = res.files
          .map((f) => f.name)
          .filter((n) => n.endsWith('.py'))
          .sort()
        setScripts(py)
      })
      .catch(() => {
        // configs 可能不存在
        setScripts([])
      })
  }, [])

  const loadRuns = useCallback(() => {
    setRunsLoading(true)
    fetchRuns('strategies')
      .then((res) => setRuns(res.runs))
      .catch(() => setRuns([]))
      .finally(() => setRunsLoading(false))
  }, [])

  useEffect(() => {
    loadScripts()
    loadRuns()
  }, [loadScripts, loadRuns])

  const loadScript = async (name: string) => {
    setSelectedScript(name)
    try {
      const data = await fetchFileContent(`configs/${name}`)
      if (data.kind === 'text') {
        setEditor(data.content)
      } else {
        message.warning('非文本文件')
        setEditor('')
      }
    } catch (e) {
      message.error(e instanceof Error ? e.message : String(e))
    }
  }

  const saveScript = async () => {
    if (!selectedScript) return
    setSaving(true)
    try {
      await putFileContent(`configs/${selectedScript}`, editor)
      message.success('已保存')
    } catch (e) {
      message.error(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const createScript = async () => {
    const name = newName.trim()
    if (!name) {
      message.warning('请输入文件名')
      return
    }
    const fname = name.endsWith('.py') ? name : `${name}.py`
    if (!/^[\w.-]+\.py$/.test(fname)) {
      message.error('文件名仅允许字母数字 _ . -')
      return
    }
    try {
      await putFileContent(`configs/${fname}`, SCRIPT_TEMPLATE)
      message.success(`已创建 configs/${fname}`)
      setNewName('')
      loadScripts()
      await loadScript(fname)
    } catch (e) {
      message.error(e instanceof Error ? e.message : String(e))
    }
  }

  const runBuiltin = async () => {
    try {
      const v = await builtinForm.validateFields()
      const argv: string[] = [
        'strategies',
        'run',
        v.name as string,
        '--start',
        v.start as string,
        '--end',
        v.end as string,
      ]
      if (v.universe) {
        argv.push('--universe', v.universe as string)
      }
      if (v.run_id) {
        argv.push('--run-id', v.run_id as string)
      }
      const sets = String(v.sets ?? '')
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean)
      for (const line of sets) {
        argv.push('--set', line)
      }
      setBuiltinSubmitting(true)
      const meta = await submitJob({
        kind: 'cli',
        argv,
        title: `strategies run ${v.name}`,
      })
      message.success(
        <span>
          已提交 {meta.job_id} · <Link to="/jobs">去任务中心</Link>
        </span>,
      )
    } catch (e) {
      if (e && typeof e === 'object' && 'errorFields' in e) return
      message.error(e instanceof Error ? e.message : String(e))
    } finally {
      setBuiltinSubmitting(false)
    }
  }

  const runScript = async () => {
    if (!selectedScript) {
      message.warning('请先选择脚本')
      return
    }
    // 先保存
    try {
      await putFileContent(`configs/${selectedScript}`, editor)
    } catch (e) {
      message.error(e instanceof Error ? e.message : String(e))
      return
    }
    const extras = scriptExtra
      .trim()
      .split(/\s+/)
      .filter(Boolean)
    const argv = [`configs/${selectedScript}`, ...extras]
    setRunning(true)
    try {
      const meta = await submitJob({
        kind: 'script',
        argv,
        title: `script ${selectedScript}`,
      })
      message.success(
        <span>
          已提交 {meta.job_id} · <Link to="/jobs">去任务中心</Link>
        </span>,
      )
    } catch (e) {
      message.error(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }

  const runColumns: ColumnsType<RunSummary> = [
    {
      title: 'run_id',
      dataIndex: 'run_id',
      key: 'run_id',
      render: (v: string) => (
        <Link to={`/run/strategies/${encodeURIComponent(v)}`}>
          <Typography.Text code>{v}</Typography.Text>
        </Link>
      ),
    },
    {
      title: 'status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (v: string | null) => v ?? '—',
    },
    {
      title: 'git_sha',
      dataIndex: 'git_sha',
      key: 'git_sha',
      width: 100,
      render: (v: string | null) => shortSha(v),
    },
  ]

  return (
    <div>
      <Typography.Title level={4} style={{ marginTop: 0 }}>
        策略回测
      </Typography.Title>
      <Typography.Paragraph type="secondary" style={{ marginTop: -8 }}>
        此页跑的是
        <Typography.Text strong>交易轨预置权重回测</Typography.Text>
        （t 日信号 t+1 执行、含约束与成本）。因子预测力评估请走命令启动器的{' '}
        <Typography.Text code>factor eval</Typography.Text>。
      </Typography.Paragraph>

      <Card size="small" title="内置策略快速跑" style={{ marginBottom: 16 }}>
        <Form
          form={builtinForm}
          layout="vertical"
          initialValues={{
            name: 'trend_timing',
            universe: 'all_a',
          }}
        >
          <Space wrap size="middle" style={{ width: '100%' }} align="start">
            <Form.Item
              name="name"
              label="策略名"
              rules={[{ required: true }]}
              style={{ minWidth: 180 }}
            >
              <Select
                options={STRATEGY_NAMES.map((n) => ({ value: n, label: n }))}
              />
            </Form.Item>
            <Form.Item
              name="start"
              label="--start"
              rules={[{ required: true, message: '必填' }]}
            >
              <Input placeholder="YYYYMMDD" style={{ width: 140 }} />
            </Form.Item>
            <Form.Item
              name="end"
              label="--end"
              rules={[{ required: true, message: '必填' }]}
            >
              <Input placeholder="YYYYMMDD" style={{ width: 140 }} />
            </Form.Item>
            <Form.Item name="universe" label="--universe">
              <Input style={{ width: 120 }} />
            </Form.Item>
            <Form.Item name="run_id" label="--run-id">
              <Input placeholder="可选" style={{ width: 160 }} />
            </Form.Item>
          </Space>
          <Form.Item
            name="sets"
            label="--set KEY=VALUE（每行一条）"
            style={{ maxWidth: 560 }}
          >
            <Input.TextArea
              rows={3}
              placeholder={'ma_window=200\ntop_n=50'}
            />
          </Form.Item>
          <Button
            type="primary"
            loading={builtinSubmitting}
            onClick={() => void runBuiltin()}
          >
            提交 strategies run
          </Button>
        </Form>
      </Card>

      <Card size="small" title="自定义策略脚本" style={{ marginBottom: 16 }}>
        <Space style={{ marginBottom: 12 }} wrap>
          <Select
            style={{ minWidth: 260 }}
            placeholder="选择 configs/*.py"
            value={selectedScript ?? undefined}
            options={scripts.map((s) => ({ value: s, label: s }))}
            onChange={(v) => void loadScript(v)}
            showSearch
          />
          <Input
            style={{ width: 180 }}
            placeholder="新建文件名.py"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <Button onClick={() => void createScript()}>新建脚本</Button>
          <Button
            disabled={!selectedScript}
            loading={saving}
            onClick={() => void saveScript()}
          >
            保存
          </Button>
          <Button
            type="primary"
            disabled={!selectedScript}
            loading={running}
            onClick={() => void runScript()}
          >
            运行
          </Button>
        </Space>
        <Input
          style={{ marginBottom: 8 }}
          placeholder="附加参数（按空格拆分，如 --start 20240101 --end 20240601）"
          value={scriptExtra}
          onChange={(e) => setScriptExtra(e.target.value)}
          disabled={!selectedScript}
        />
        <Input.TextArea
          value={editor}
          onChange={(e) => setEditor(e.target.value)}
          disabled={!selectedScript}
          autoSize={{ minRows: 14, maxRows: 32 }}
          style={{
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: 13,
            width: '100%',
          }}
          placeholder="选择或新建脚本后在此编辑"
        />
      </Card>

      <Card
        size="small"
        title={
          <Space>
            <span>策略回测产物</span>
            <Typography.Text type="secondary" style={{ fontWeight: 400 }}>
              workspace/strategies/ ·{' '}
              <Link to="/domain/strategies">完整列表</Link>
            </Typography.Text>
          </Space>
        }
        extra={
          <Button size="small" onClick={() => loadRuns()}>
            刷新
          </Button>
        }
      >
        <Table
          rowKey="run_id"
          size="small"
          loading={runsLoading}
          columns={runColumns}
          dataSource={runs.slice().reverse()}
          pagination={{ pageSize: 10 }}
          locale={{ emptyText: '暂无 strategies 产物' }}
        />
      </Card>
    </div>
  )
}
