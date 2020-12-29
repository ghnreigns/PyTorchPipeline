"""Some accuracy meters."""
import sklearn
from sklearn.metrics import accuracy_score


class AverageLossMeter:
    """
    Computes and stores the average and current loss
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.curr_batch_avg_loss = 0
        self.avg = 0
        self.running_total_loss = 0
        self.count = 0

    def update(self, curr_batch_avg_loss: float, batch_size: str):
        self.curr_batch_avg_loss = curr_batch_avg_loss
        self.running_total_loss += curr_batch_avg_loss * batch_size
        self.count += batch_size
        self.avg = self.running_total_loss / self.count


class AccuracyMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0
        self.count = 0
        self.sum = 0

    def update(self, y_true, y_pred, batch_size=1):

        # so we just need to count total num of images / batch_size
        # self.count += num_steps
        self.batch_size = batch_size
        self.count += self.batch_size
        # this part here already got an acc score for the 4 images, so no need divide batch size
        self.score = sklearn.metrics.accuracy_score(y_true, y_pred)
        total_score = self.score * self.batch_size

        self.sum += total_score

    @property
    def avg(self):
        self.avg_score = self.sum / self.count
        return self.avg_score
