import logging
import torch
import numpy as np
from typing import Optional, Union
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers.modeling_outputs import BaseModelOutput

DEFAULT_HUGGINGFACE_LENGTH = 512
logger = logging.getLogger(__name__)


# Fix bug in ORTModelForFeatureExtraction, see https://github.com/huggingface/optimum/pull/1941/files
class MyORTModelForFeatureExtraction(ORTModelForFeatureExtraction):
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_ids,
                attention_mask,
                token_type_ids,
                ordered_input_names=self._ordered_input_names,
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(
                output_shapes["last_hidden_state"]
            )
        else:
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }

            onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

            if "last_hidden_state" in self.output_names:
                last_hidden_state = model_outputs["last_hidden_state"]
            else:
                # TODO: This allows to support sentence-transformers models (sentence embedding), but is not validated.
                last_hidden_state = next(iter(model_outputs.values()))

        # converts output to namedtuple for pipelines post-processing
        return BaseModelOutput(last_hidden_state=last_hidden_state)
