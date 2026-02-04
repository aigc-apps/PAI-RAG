// app/evaluation/[evalId]/types.ts

export interface EvalConfig {
    id: string;
    name: string;
    description: string;
    type: string;
}

export interface SampleItem {
    id: string;
    input: string;
    expected_output: string;
    eval_metadata?: {
        Steps?: string;
        Tools?: string;
        file_name?: string;
    };
}

export interface RunConfig {
    id: string;
    name: string;
    model_id: string;
    mcp_ids: string[];
    kb_ids: string[];
    enable_search: boolean;
    enable_vision: boolean;
    enable_agent: boolean;
    enable_input_guardrail?: boolean;
    enable_output_guardrail?: boolean;
    guardrail_hint?: string;
    prompts: {
      react: string;
    };
}

export interface EvaluatorConfig {
    id: string;
    name: string;
    type: string;
    model_id: string;    
    case_sensitive?: boolean;
    ignore_punctuation?: boolean;
}

export interface ExperimentItem {
  id: string;
  samples_count: number;
  name: string;
  description: string;
  status: string;
  run_config_id: string;
  evaluator_config_id: string;
  avg_score: number;
  created_at: string;
  updated_at: string;
}

export interface ExperimentSampleDetails {
  id: string
  experiment_id: string
  dataset_id: string
  sample_id: string
  input: string
  expected_output: string
  actual_output: string
  trace_id: string
  status: string
  score: number
  reason: string
  dataset_metadata?: {
    Steps?: string
    Tools?: string
  }
  execution_metadata?: {
    id: string
    index: number
    function: {
      name: string
      arguments: string
    }
    type: string
    observation: {
      result: string | null
    }
  }[]
  created_at: string
  started_at: string | null
  updated_at: string
}