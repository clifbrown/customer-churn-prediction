"""Execute the notebook sequentially in an in-process IPython session.

This avoids a separate Jupyter kernel/socket requirement. Real stdout and rich
IPython display outputs are captured into the notebook; any cell error aborts
execution. No previous cell outputs are reused. The notebook also works in a
normal Jupyter kernel. Run from the repository root.
"""
from pathlib import Path
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    path=ROOT/'notebooks/churn_eda_modeling.ipynb'
    nb=nbformat.read(path,as_version=4)
    shell=InteractiveShell.instance()
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        count+=1
        cell.outputs=[]
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(cell.source,store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Notebook cell {count} failed: {captured.stdout}\n{captured.stderr}')
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
        cell.execution_count=count
        print(f'Executed code cell {count}',flush=True)
    nb.metadata['execution']={'method':'in-process IPython; real stdout and rich display capture',
                              'code_cells_executed':count,'errors':0}
    nbformat.validate(nb)
    nbformat.write(nb,path)
    print(f'Notebook saved: {count} code cells executed with zero errors.')


if __name__=='__main__':main()
