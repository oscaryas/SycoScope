import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.data import dataset_control as dc


class DatasetControlTests(unittest.TestCase):
    def test_crossed_conditions_and_neutral(self):
        exp={'pairs':[{'slug':'pair','sycophantic':'positive','non_sycophantic':'negative'}],
             'prompts':[{'prompt_id':f'{d}-{s}','dataset':d,'split':s,'user_prompt':'Question'}
                        for d in ('perez','dolly') for s in ('train','report')]}
        paired=dc.work_items(exp,'pair');neutral=dc.work_items(exp,'neutral')
        self.assertEqual(len(paired),8)
        self.assertEqual(len({r['example_id'] for r in paired+neutral}),12)
        for prompt in exp['prompts']:
            rows=[r for r in paired if r['prompt_id']==prompt['prompt_id']]
            self.assertEqual({r['label'] for r in rows},{0,1})
            self.assertEqual({r['split'] for r in rows},{prompt['split']})
        self.assertTrue(all(r['label'] is None and r['system_prompt']=='' for r in neutral))

    def test_manifest_is_immutable_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest.json'
            dc.save_new(path,{'seed':1})
            dc.save_new(path,{'seed':1})
            with self.assertRaises(ValueError):dc.save_new(path,{'seed':2})
            self.assertEqual(json.loads(path.read_text()),{'seed':1})

    def test_normalization_and_digest(self):
        self.assertEqual(dc.normalize(' A  question\nHere '),'a question here')
        self.assertEqual(dc.digest({'a':1,'b':2}),dc.digest({'b':2,'a':1}))


if __name__=='__main__':unittest.main()
