# OVER 50 mb do not include in git push

find . -type f -size +50M > large_files.txt

sed -i 's/^\.\///' large_files.txt
sed -i 's|^|/scratch/user/dylanelliott/letkf-hybrid-speedy-grace-training-edition-original/|' large_files.txt

# remove from git index 
while IFS= read -r file; do
    git rm --cached "$file"
done < large_files.txt

echo "DONE"
