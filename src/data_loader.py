import os
import pandas as pd
import numpy as np
from datetime import datetime
import re

IMS_TEST_INFO = {
    '1st_test': {
        'set_number': 1,
        'channels': 8,
        'channel_names': ['Bearing1_X', 'Bearing1_Y', 'Bearing2_X', 'Bearing2_Y',
                          'Bearing3_X', 'Bearing3_Y', 'Bearing4_X', 'Bearing4_Y'],
        'known_failures': {
            'Bearing3': 'Inner race defect',
            'Bearing4': 'Roller element defect'
        },
        'description': 'Test 1: Oct 2003 - Nov 2003. Two accelerometers per bearing (X and Y axes).'
    },
    '2nd_test': {
        'set_number': 2,
        'channels': 4,
        'channel_names': ['Bearing1', 'Bearing2', 'Bearing3', 'Bearing4'],
        'known_failures': {
            'Bearing1': 'Outer race defect'
        },
        'description': 'Test 2: Feb 2004. One accelerometer per bearing.'
    },
    '4th_test': {
        'set_number': 3,
        'channels': 4,
        'channel_names': ['Bearing1', 'Bearing2', 'Bearing3', 'Bearing4'],
        'known_failures': {
            'Bearing3': 'Outer race defect'
        },
        'description': 'Test 3 (Set No. 3): Mar 2004 - Apr 2004. One accelerometer per bearing.'
    }
}

class IMSDataLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_test_info(self):
        return IMS_TEST_INFO

    def get_available_tests(self):
        if not os.path.exists(self.base_path):
            return []
        tests = []
        for item in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, item)) and item in IMS_TEST_INFO:
                tests.append(item)
        return tests

    def get_file_list(self, test_name):
        if test_name not in IMS_TEST_INFO:
            raise ValueError(f"Unknown test name: {test_name}")
        
        test_dir = os.path.join(self.base_path, test_name)
        if test_name == '4th_test':
            test_dir = os.path.join(test_dir, 'txt')
            
        if not os.path.exists(test_dir):
            return []
            
        files = []
        for f in os.listdir(test_dir):
            if test_name == '2nd_test' and f == '.ipynb_checkpoints':
                continue
            file_path = os.path.join(test_dir, f)
            if os.path.isfile(file_path):
                files.append(file_path)
                
        # Sort by timestamp embedded in filename
        def parse_filename_sort(filepath):
            basename = os.path.basename(filepath)
            # Extract timestamp part - handle both short and long filenames
            match = re.search(r'(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2})', basename)
            if match:
                return match.group(1)
            return basename
        
        return sorted(files, key=parse_filename_sort)

    def parse_timestamp(self, filename):
        # Filename format: '2003.10.22.12.06.24'
        try:
            return datetime.strptime(filename, '%Y.%m.%d.%H.%M.%S')
        except ValueError:
            # Fallback if there are extra characters
            match = re.search(r'(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2})', filename)
            if match:
                return datetime.strptime(match.group(1), '%Y.%m.%d.%H.%M.%S')
            return None

    def load_file(self, filepath, test_name=None):
        if test_name is None:
            # Auto-detect from path
            if '1st_test' in filepath: test_name = '1st_test'
            elif '2nd_test' in filepath: test_name = '2nd_test'
            elif '4th_test' in filepath: test_name = '4th_test'
            else: raise ValueError("Could not auto-detect test_name from filepath")

        cols = self.get_channel_names(test_name)
        # Use tab separator, no header
        df = pd.read_csv(filepath, sep='\t', header=None)
        # Slice to match known channels - handle both fewer and extra columns
        if df.shape[1] >= len(cols):
            df = df.iloc[:, :len(cols)]
        else:
            # If fewer columns, pad with NaN
            df = df.iloc[:, :df.shape[1]]
            for c in range(df.shape[1], len(cols)):
                df[c] = np.nan
        df.columns = cols
        return df

    def load_multiple_files(self, test_name, file_indices=None, max_files=None):
        file_list = self.get_file_list(test_name)
        if file_indices:
            file_list = [file_list[i] for i in file_indices if i < len(file_list)]
        if max_files and len(file_list) > max_files:
            file_list = file_list[:max_files]
            
        results = []
        for filepath in file_list:
            filename = os.path.basename(filepath)
            timestamp = self.parse_timestamp(filename)
            df = self.load_file(filepath, test_name)
            results.append((filepath, timestamp, df))
        return results

    def get_channel_names(self, test_name):
        if test_name in IMS_TEST_INFO:
            return IMS_TEST_INFO[test_name]['channel_names']
        return []

    def get_test_metadata(self, test_name):
        if test_name not in IMS_TEST_INFO:
            return {}
        
        info = IMS_TEST_INFO[test_name]
        files = self.get_file_list(test_name)
        
        return {
            'test_name': test_name,
            'num_files': len(files),
            'num_channels': info['channels'],
            'sampling_rate': 20480,
            'samples_per_file': 20480,
            'duration_seconds': 1.0,
            'recording_interval': '~10 minutes',
            'channel_names': info['channel_names'],
            'known_failures': info['known_failures'],
            'bearing_type': 'Rexnord ZA-2115',
            'rpm': 2000,
            'load': '6000 lbs'
        }
