"""Package exact E3 filenames. This does not submit anything externally."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json
import shutil

def main():
    team=json.loads(Path('Paper/team.json').read_text(encoding='utf-8-sig'))['team_id']
    for suffix in ['report.pdf','report_anonymized.pdf']:
        shutil.copyfile(f'Paper/HW2_{team}_{suffix}',f'HW2_{team}_{suffix}')
    names=[f'HW2_{team}_report.pdf',f'HW2_{team}_presentation.pptx']
    with ZipFile(f'HW2_{team}.zip','w',compression=ZIP_DEFLATED) as out:
        for name in names: out.write(name,arcname=name)
    print(f'HW2_{team}.zip contains exactly: '+', '.join(names))

if __name__=='__main__': main()
