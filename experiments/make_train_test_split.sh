#!/usr/bin/env bash

# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Convenience script for making train and test sets of .gml files out of a bunch of java (maven) repositories
# Splits the repositories into "seen" and "unseen", then aggregates gml files from each category into train and test sets
# You'll be left with a seen_repos directory and an unseen_repos directory containing said repos, and
# with 3 directories full of .gml files:
#   seen_repos/train_graphs
#   seen_repos/test_graphs
#   unseen_repos/test_graphs
#
# This script must be run from the top level of a directory containing all the repos in a directory called "repositories"

rm -rf ./temp
rm -rf ./seen_repos
rm -rf ./unseen_repos

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}
seed=42

# Split into seen and unseen repos
mkdir seen_repos
mkdir unseen_repos
mkdir temp

num_repos=$(ls ./repositories | wc -l)
echo $num_repos total repos
echo $(find ./repositories -type f -name "*.gml" | wc -l) total files

unseen_size=$((num_repos * 20 / 100))
echo Separating $unseen_size unseen repos

ls -d -1 ./repositories/* | xargs -I{} cp -r {} ./temp/
ls -d -1 ./temp/* | shuf --random-source=<(get_seeded_random $seed) | tail -$unseen_size | xargs -I{} mv {} unseen_repos
ls -d -1 ./temp/* | xargs -I{} mv {} seen_repos
rmdir ./temp
echo Repos split

# Split files into train and test
mkdir unseen_repos/test_graphs
find ./unseen_repos/ -type f -name "*.gml" | xargs -I{} mv {} ./unseen_repos/test_graphs
echo $(ls -d -1 ./unseen_repos/test_graphs/* | wc -l) total unseen test graphs

mkdir seen_repos/train_graphs
mkdir seen_repos/test_graphs
mkdir seen_repos/temp

seen_files=$(find ./seen_repos/ -type f -name "*.gml")
mv $seen_files ./seen_repos/temp

num_files=$(ls ./seen_repos/temp | wc -l)
echo $num_files total seen files

test_size=$((num_files * 15 / 100))
echo $test_size in test set

ls -d -1 ./seen_repos/temp/* | shuf --random-source=<(get_seeded_random $seed) | tail -$test_size | xargs -I{} mv {} ./seen_repos/test_graphs
ls -d -1 ./seen_repos/temp/* | xargs -I{} mv {} ./seen_repos/train_graphs
rmdir seen_repos/temp
