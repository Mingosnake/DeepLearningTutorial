## How to train
```
$ ./train.py --config config/CONFIG_NAME.yaml
```
## How to test
```
$ ./test.py --result result/CONFIG_NAME/EXP_VERSION/
```
## How to open tensorboard
```
$ tensorboard --logdir=PROJECT_PATH/result/ --host 0.0.0.0 --port PORT_NUMBER
```
And, open `http://SERVER_URL:PORT_NUMBER` in browser.
