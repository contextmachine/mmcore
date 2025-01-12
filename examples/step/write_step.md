# Writing STEP files example


mmcore provides some (as yet limited) step file writing capabilities. At the moment it is possible to write one or more NURBSSurface objects to a single step file. 

> Topological entities are currently in an initial state and are obviously not supported for writing.

Для того чтобы сгенерировать пример step файла запускайте examples/step/write.py

To generate an example step file run `examples/step/write.py`

For details, read the source code of the [examples/step/write.py](../../examples/step/write.py) file and the [`mmcore.compat.step.step_writer`](../../mmcore/compat/step/step_writer.py) module