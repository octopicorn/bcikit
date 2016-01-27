Sample usage (assuming you're in cloudbrain root dir):
python cloudbrain/apps/Analysis/AnalysisService.py -i octopicorn -c localhost -n openbci


The basic idea of Analysis is that each module has input/output and the modules are chained in order
with one module's output going into the other's input.

Some assumptions:

- so far, this only works with Pika, not Pipe
- so far this has only been tested with OpenBCI on the "eeg" metric
- the module processing chain is configured in conf.yml
- each module key "class" in yaml file must correspond to the exact same class in some file {class}.py 
Analysis/modules folder
- each module in yml file has inputs, outputs, and parameters.
- standard item in parameters per module is "debug".  If True, it governs output to stdout (via print commands)
- modules are loaded, in order, by AnalysisService.py.
- each module is run in its own asynchronous, nonblocking thread
- semantically, "metric" and "feature" are interchangeable terms
- Ctrl-C shuts the whole processing chain down, and all the child threads. this is possible because the threads run in
daemon mode, and also because the parent process AnalysisService::start has an endless loop running which
can be interrupted.

