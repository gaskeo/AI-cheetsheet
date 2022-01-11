# callbacks
Коллбэки нужны для отслеживания состояния модели во время обучения

```python
import tensorflow as tf

class MyCallBack(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs): 
        if logs['loss'] < 90: 
            print('конец') 
            self.model.stop_training = True


model = ...
...
model.fit(..., callbacks=[MyCallBack()])
```