import tempfile
import unittest
from pathlib import Path
import numpy as np
from dagger import DAgger
from partial_observation import PartialObservationDAgger, corrupt_image


class ObservationChecks(unittest.TestCase):
    def test_corruption_properties(self):
        image = np.ones(128)
        np.testing.assert_array_equal(corrupt_image(image), image)
        np.testing.assert_array_equal(corrupt_image(image, mask_rate=1, noise_std=.2), np.zeros(128))
        a = corrupt_image(image, mask_rate=.3, seed=4)
        b = corrupt_image(image, mask_rate=.6, seed=4)
        self.assertTrue(np.all(b[a == 0] == 0))
        np.testing.assert_array_equal(a, corrupt_image(image, mask_rate=.3, seed=4))
        self.assertFalse(np.array_equal(a, corrupt_image(image, mask_rate=.3, seed=5)))
        noise = corrupt_image(image*.5, noise_std=.2)
        self.assertTrue(np.all((noise >= 0) & (noise <= 1)))
        self.assertFalse(np.array_equal(noise,image*.5))
        np.testing.assert_array_equal(image,np.ones(128))
        for kwargs in [dict(mask_rate=-.1),dict(mask_rate=1.1),dict(noise_std=-1),dict(noise_std=float('nan'))]:
            with self.assertRaises(ValueError): corrupt_image(image,**kwargs)

    def test_clean_equivalence_and_fixed_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'tiny.data'
            rows=[]
            for fold in range(10):
                for label in range(2):
                    rows.append('\t'.join([str(len(rows)),'ab'[label],'-1','0','0',str(fold)]+[str(label)]*128))
            source.write_text('\n'.join(rows))
            original=DAgger(str(source))
            clean=PartialObservationDAgger(str(source))
            a,models=original.run(N=2,plot=False,output_dir=tmp)
            b,others=clean.run(N=2,plot=False,output_dir=tmp)
            np.testing.assert_array_equal(a,b)
            for p,q in zip(models,others): np.testing.assert_array_equal(p.coef_,q.coef_)
            masked=PartialObservationDAgger(str(source),mask_rate=.5,observation_seed=3)
            masked.build_initial_dataset()
            saved=[word[0].copy() for word in masked.words]
            masked.process_ocr()
            self.assertEqual(masked.sequences,original.sequences)
            self.assertEqual(masked.words_fold,original.words_fold)
            for old,word in zip(saved,masked.words): np.testing.assert_array_equal(old,word[0])
            masked.run(N=2,plot=False,output_dir=tmp)
            self.assertTrue((masked.last_run_dir/'config.json').exists())


if __name__ == '__main__': unittest.main()
