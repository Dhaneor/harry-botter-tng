#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 06 10:03:20 2021

@author dhaneor
"""
import os, sys
import re
import shutil

# -----------------------------------------------------------------------------
wd = os.curdir + '/src/exchange/util'
print(f'using directory: {wd}\n')
os.chdir(wd)    

# specifiy the source directory here. this directory must
# contain the kucoin-python-sdk files
source_dir = './kucoin_sync'
dest_dir = os.curdir + '/kucoin_async'

print(source_dir, dest_dir)
shutil.copytree(source_dir, dest_dir)

# find all files that we need to change
candidates = []
search = ['trade.py', 'margin.py', 'market.py', 'user.py', 'token.py']
# Traversing through Test 
for root, dirs, files in os.walk('kucoin_sync'): 
    for d in dirs:
        candidates += list(set([os.path.join(root, f) for f in files \
                for term in search if term in f and not f.endswith('.pyc')]))
    
[print(c) for c in candidates]

# replace all synchronous method definitions and include await
# where necessary
for file in candidates:
    with open(file, "r") as f:
        lines = f.readlines()
        
    output = []
    
    for line in lines:
        if re.search(
            "from kucoin.base_request.base_request import KucoinBaseRestApi",
            line
        ):
            line = re.sub(
                "from kucoin.base_request.base_request import KucoinBaseRestApi",
                "from ..kucoin_async_client import AsyncKucoinBaseRestApi",
                line
            )
            # print(line)
        # from kucoin_async_client import AsyncKucoinBaseRestApi
        
        if '(KucoinBaseRestApi)' in line:
            line = line.replace(
                '(KucoinBaseRestApi)', 
                '(AsyncKucoinBaseRestApi)'
            )
            
        if re.search(" def ", line):
            line = re.sub(" def ", " async def ", line)
            # print(f'{file} >>>', line)
        
        if re.search(" return self._request", line):
            line = re.sub(" return self._request", " return await self._request", line)
            # print(f'{file} >>>', line) 
            
        output.append(line)
            

    # write the modified code to the file
    split = file.split('.')
    if not re.search("new\Z", split[0]):     #type:ignore 

        # switch these lines for test runs
        # newfile = f'{split[0]}_new.{split[1]}'
        newfile_path = '/'.join(file.split('/')[1:])
        newfile = f'{dest_dir}/{newfile_path}'
        print(newfile)

        with open(newfile, "w") as f:
            for line in output:
                f.writelines(line) #+'\n')

# copy the modified base client to the destination                
async_client_file = os.curdir + '/kucoin_async_client.py'
dest_dir = os.curdir + '/kucoin_async'
shutil.copy(async_client_file, dest_dir)

                
            

