import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from .gpe_timeseries import list_variables, get_series, get_metric

@dataclass
class ObservableHandler:
    name: str
    variable: Optional[str] = None
    metric: str = "max"
    func: Optional[Callable] = None

    def evaluate(self, results):
        if self.func is not None:
            return float(self.func(results))

        y, _ = get_series(results, self.variable)
        
        if y.size == 0:
            raise KeyError(f"Variable {self.variable} not found in results")
        
        return get_metric(y,self.metric)

'''
Obtain pressure and flow values for various block types 
'''
def get_observables(results, fields=("pressure","flow"), metrics=("max","mean")):
    observables = []

    for var in list_variables(results):
        field = var.split(":", 1)[0]

        if field in fields:
            for m in metrics:
                observables.append(ObservableHandler(name=f"{var}|{m}",variable=var,metric=m))

    return observables

'''
Function to obtain volume metrics (EDV, ESV, Stroke volume, Ejection fraction)
'''
def get_volumes(volume_observable, volume_baseline=0.0):
    def volume(results):
        vol, _ = get_series(results, volume_observable)

        if vol.size == 0:
            raise KeyError(f"Volume variable {volume_observable} not found in results dataframe")

        return np.asanyarray(vol) + volume_baseline

    def edv(results):
            return float(np.max(volume(results)))
    
    def esv(results):
        return float(np.min(volume(results)))

    def stroke(results):
        return edv(results) - esv(results)

    def ef(results):
        return (stroke(results)) / edv(results)

    edv_observable = ObservableHandler(name="edv", func=edv)
    esv_observable = ObservableHandler(name="esv", func=esv)
    stroke_observable = ObservableHandler(name="stroke", func=stroke)
    ef_observable = ObservableHandler(name="ef", func=ef)

    return edv_observable, esv_observable, stroke_observable, ef_observable
