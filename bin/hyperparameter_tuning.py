from neo_tracklet_classifier.datapipeline  import DataPipeline
from tensorflow import data, nn, summary
from tensorflow import summary
from keras import layers, optimizers, metrics, models, regularizers
from tensorboard.plugins.hparams import api as hp

TOTAL_SAMPLES = 5997886
SUB_SAMPLE_SIZE = 5997886
TRAINING_SIZE = SUB_SAMPLE_SIZE * 0.9
DEVIATION = 0.05
PATIENCE = 10
EPOCHS = 10000
BATCH_SIZE = 1024


HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
HP_LREG = hp.HParam('reg', hp.RealInterval(1e-6, 1e-4))
HP_NUM_UNITS_L1 = hp.HParam('num_neurons1', hp.Discrete([512, 1024]))
HP_NUM_UNITS_L2 = hp.HParam('num_neurons2', hp.Discrete([256, 512, 1024]))
METRIC_PRC = 'prc'  # precision-recall curve

def train_test_model(hparams):
  model = models.Sequential([
    layers.Dense(hparams[HP_NUM_UNITS_L1], activation=nn.relu, kernel_regularizer=regularizers.l2(hparams[HP_LREG])),
    layers.Dropout(hparams[HP_DROPOUT]),
    layers.Dense(hparams[HP_NUM_UNITS_L2], activation=nn.relu, kernel_regularizer=regularizers.l2(hparams[HP_LREG])),
    layers.Dropout(hparams[HP_DROPOUT]),
    layers.Dense(1, activation=nn.sigmoid),
  ])
  model.compile(
      optimizer=optimizers.Adam(),
      loss='binary_crossentropy',
      metrics=[metrics.AUC(name='prc', curve='PR')],
  )
  
  model.fit(
    data.train_ds,
    epochs=1,
    batch_size=BATCH_SIZE) # Run with 1 epoch to speed things up for demo purposes
  
  print(model.metrics_names)
  _,prc = model.evaluate(data.test_ds)
  
  return prc

def run(run_dir, hparams):
  with summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    prc = train_test_model(hparams)
    summary.scalar(METRIC_PRC, prc, step=1)

    
if __name__ == "__main__":

    data = DataPipeline.DataPipeline(
        TOTAL_SAMPLES, SUB_SAMPLE_SIZE, 
        TRAINING_SIZE, 1024, 
        DEVIATION, ragged=False)

    session_num = 0
    
    with summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS_L1, HP_NUM_UNITS_L2, HP_DROPOUT, HP_LREG],
            metrics=[hp.Metric(METRIC_PRC, display_name='PRC')])
       
       
    for num_units1 in HP_NUM_UNITS_L1.domain.values:
      for num_units2 in HP_NUM_UNITS_L2.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value/2, HP_DROPOUT.domain.max_value):
          for l_reg in (HP_LREG.domain.min_value, HP_LREG.domain.min_value*10, HP_LREG.domain.max_value):
            hparams = {
                HP_NUM_UNITS_L1: num_units1,
                HP_NUM_UNITS_L2: num_units2,
                HP_DROPOUT: dropout_rate,
                HP_LREG: l_reg
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + run_name, hparams)
            session_num += 1