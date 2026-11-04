"""Export predeclared seed-1 BCE/ordinal heads; no test-based model selection."""
import argparse
import json
from pathlib import Path

import torch

from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.inference import LoadingEstimator
from himloco_lab.impact.models import ModelBank,outputs_and_losses


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('training',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    spec=json.loads((args.training/'spec.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    exports=[]
    for task in spec['tasks']:
        bank=ModelBank(spec['models'],spec['training_seeds'],spec['hidden_sizes']).eval()
        saved=torch.load(args.training/task/'models.pt',map_location='cpu',weights_only=True)
        bank.load_state_dict(saved['model'])
        calibration=json.loads((args.training/task/'calibration.json').read_text())
        generator=torch.Generator().manual_seed(391)
        x=torch.randn(32,270,generator=generator)
        with torch.no_grad():
            scores=outputs_and_losses(bank(x,torch.arange(len(x))),torch.zeros(32,4),
                                      bank.entries,torch.tensor([20.,40.,80.]))[0]
        for method in ('bce','ordinal'):
            index=bank.entries.index((1,method))
            module=LoadingEstimator(bank,index,calibration[index]).eval()
            scripted=torch.jit.script(module)
            expected=torch.sigmoid(scores[index]*calibration[index]['slope']+calibration[index]['intercept'])
            with torch.no_grad():
                actual=scripted(x)
            error=float((actual['primary_probability']-expected).abs().max())
            if error>1e-5:
                raise AssertionError(f'Export disagrees with model bank: {error}')
            ps=actual['exceedance_probabilities']
            if method=='ordinal' and not torch.all(ps[...,:-1]>=ps[...,1:]):
                raise AssertionError('Calibration broke severity ordering')
            path=args.output/f'{task}_{method}_seed1.pt'
            scripted.save(str(path))
            exports.append({'task':task,'method':method,'seed':1,'file':path.name,'sha256':file_hash(path),
                            'max_model_bank_error':error,'thresholds_n':module.thresholds_n.tolist(),
                            'selection':'predeclared example; not a test-selected winner'})
    (args.output/'manifest.json').write_text(json.dumps({'label':spec['label'],'exports':exports,
          'input':'current frame first followed by five causal past frames, 45D per frame',
          'calibration':'same increasing transform applied to all ordinal logits; fit at 40N only',
          'limitation':'simulation loading proxy; not validated for robot control'},indent=2)+'\n')
    print(json.dumps(exports,indent=2))
