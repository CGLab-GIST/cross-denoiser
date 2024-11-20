import torch

class TimeRecord:
    def __init__(self) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start(self):
        torch.cuda.synchronize()
        self.start_event.record()
    
    def end(self):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed = self.start_event.elapsed_time(self.end_event)
        return elapsed


class Timing:
    def __init__(self) -> None:
        self.records = {}
        self.record_time = {}
        self.record_num = {}
        self.last_elapsed = {}
    
    def add(self, name):
        record = TimeRecord()
        self.records[name] = record
        self.record_time[name] = 0
        self.record_num[name] = 0
        self.last_elapsed[name] = 0
    
    def start(self, name):
        if name not in self.records:
            self.add(name)
        self.records[name].start()
    
    def end(self, name, exclude_from_time=False):
        if name not in self.records:
            print(f'Cannot .end(). Time record {name} does not exist!')
            return

        elapsed = self.records[name].end()
        if not exclude_from_time:
            self.record_time[name] += elapsed
            self.record_num[name] += 1
            self.last_elapsed[name] = elapsed
        
    def to_string(self, name=None, total_only=False):
        if name == None:
            ret = ''
            if total_only == False:
                for i, rec in enumerate(self.records):
                    if i > 0:
                        ret += ', '
                    ret += self.to_string(rec)
                if len(self.records) > 0:
                    ret += ', '
            # ret += f'Total {self.get_total_avg():.2f} ms / {self.get_total_time():.2f}ms'
            ret += f'Total {self.get_total_avg():.2f} ms'
        else:
            elapsed = self.last_elapsed[name]
            time = self.record_time[name]
            avg = time / self.record_num[name] if self.record_num[name] > 0 else 0
            ret = f'{name} {avg:.2f} ms'
        return ret
    
    def get_total_elapsed(self):
        total = 0
        for t in self.last_elapsed.values():
            total += t
        return total

    def get_total_time(self):
        total = 0
        for t in self.record_time.values():
            total += t
        return total

    def get_total_avg(self):
        total = 0
        num = 0
        for name, t in self.record_time.items():
            total += t
            num = self.record_num[name]
        if num == 0:
            return 0
        else:
            return total / num