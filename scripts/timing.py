"""
Copyright 2024 CGLab, GIST.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import tensorflow as tf
import time

class TFTimeRecord:
    def __init__(self) -> None:
        self.start_time = 0
        self.end_time = 0
    
    def start(self):
        # Sync before starting
        tf.test.experimental.sync_devices()
        self.start_time = time.perf_counter()
    
    def end(self):
        # Sync before ending
        tf.test.experimental.sync_devices()
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        return elapsed


class Timing:
    def __init__(self) -> None:
        self.records = {}
        self.record_time = {}
        self.record_num = {}
        self.last_elapsed = {}
    
    def add(self, name):
        record = TFTimeRecord()
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
            # ret += f'Total {self.get_total_avg()*1000:.2f} ms / {self.get_total_time()*1000:.2f}ms'
            ret += f'Total {self.get_total_avg()*1000:.2f} ms'
        else:
            elapsed = self.last_elapsed[name]
            time = self.record_time[name]
            avg = time / self.record_num[name] if self.record_num[name] > 0 else 0
            ret = f'{name} {avg*1000:.2f} ms'
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