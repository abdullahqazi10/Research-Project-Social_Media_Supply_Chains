#!"C:\Users\abdul\OneDrive\Desktop\Computer Science\Research Project\venv\Scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'twitterscraper==1.6.1','console_scripts','twitterscraper'
__requires__ = 'twitterscraper==1.6.1'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('twitterscraper==1.6.1', 'console_scripts', 'twitterscraper')()
    )
