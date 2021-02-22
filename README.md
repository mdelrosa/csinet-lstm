# CsiNet-LSTM

Repository for reproduction of CsiNet-LSTM used in MarkovNet paper (currently preprint).

- [1] Liu, Z., del Rosario, M., & Ding, Z. (2020). A Markovian Model-Driven Deep Learning Framework for Massive MIMO CSI Feedback. arXiv preprint arXiv:2009.09468.
- [2] T. Wang, C. Wen, S. Jin, and G. Y. Li, “Deep learning-based csi feedback approach for time-varying massive mimo channels,” IEEE Wireless Communications Letters, vol. 8, no. 2, pp. 416–419, April 2019.

## Data

**TODO**: Add a link to 20 timeslot COST2100 data. 

## Dependencies

**TODO**: Add a) exhaustive list of dependencies for this repo and/or b) .yml/.dockerfile for setting up working environment.

This repository relies heavily on the [`brat` repository](https://github.com/mdelrosa/brat). In particular, the `brat/utils` directory handles parsing .json config files, loading data, and evaluating network performance.

The typical hierarchy which this 

- `home` (your home directory)
    - `git`
        - `brat` (repo available [here](https://github.com/mdelrosa/brat))
        - `csinet-lstm` (this repo)
            - `csinet-lstm`
                - `csinet_train.py` (training script for csinet at single timeslot)
                - `csinet_lstm_train.py` (training script for csinet_lstm; typically ten timeslots)
