import logging
import time

logger = logging.getLogger('root')


class Progbar(object):

    def __init__(self, loader, metrics, niter=1, name: str = 'train'):
        self.num_iterations = len(loader)
        self.metrics = metrics
        self.alpha = 0.98
        self.niter = niter
        self.start = time.time()
        self.average_time = 0
        self.previous_time = self.start
        self.name = name

    def __call__(self, engine):
        num_seen = engine.state.iteration - self.num_iterations * (engine.state.epoch - 1)

        if num_seen % self.niter == 0:
            for k, v in engine.state.metrics.items():
                self.metrics[k] = v

            percent_seen = 100 * float(num_seen) / self.num_iterations
            equal_to = int(percent_seen / 10)
            done = int(percent_seen) == 100

            #total_time = self.average_time * self.num_iterations / self.niter
            
            if self.average_time == 0:
                self.average_time = time.time() - self.previous_time
            else:
                self.average_time = self.average_time * self.alpha + \
                                    (1 - self.alpha) * (time.time() - self.previous_time)

            # timer = " [{:>3.0f}<{:>3.0f}, {:>3.2f} s/it]" \
            #     .format(time.time() - self.start, total_time, self.average_time / self.niter)

            bar = '[' + '=' * equal_to + '>' * (not done) + ' ' * (10 - equal_to) + ']'

            message = '{name} | Epoch {epoch} | {percent_seen:>2.0f}% | {bar}'.format(
                name=self.name,
                epoch=engine.state.epoch,
                percent_seen=percent_seen,
                bar=bar)

            for key, value in self.metrics.items():
                message += ' | {name}: {value:.3f}'.format(name=key, value=value)

            self.previous_time = time.time()
            logger.log(32, message)

        if num_seen > self.num_iterations - 1:
            self.start = self.previous_time
