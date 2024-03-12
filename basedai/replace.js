const fs = require('fs');
const path = require('path');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const searchDirectory = './'; // Set your starting directory here
const defaultPattern = 'basedaiscan.com'; // Set your default pattern to search for
const defaultReplacement = 'basedaiscan.com'; // Set your default replacement value
const ignoreFileNames = ['CHANGELOG.md', '.github', '.git', 'node_modules', 'dist']; // Set the list of file or directory names to ignore

// Function to adjust the replacement value's case to match the original's case
const adjustCase = (original, replacement) => {
  // Check if the original is all uppercase
  if (original.toUpperCase() === original) return replacement.toUpperCase();
  // Check if the original is capitalized (first letter uppercase, rest lowercase)
  if (original.charAt(0).toUpperCase() + original.slice(1).toLowerCase() === original) {
    return replacement.charAt(0).toUpperCase() + replacement.slice(1).toLowerCase();
  }
  // Check if the original is all lowercase
  if (original.toLowerCase() === original) return replacement.toLowerCase();
  // Default to original replacement if mixed case or other
  return replacement;
};

const confirmAndReplaceInFile = (filePath, searchValue, replaceValue, fileContent, callback) => {
  const regex = new RegExp(searchValue, 'gi');
  let match;

  const replaceNextMatch = () => {
    match = regex.exec(fileContent);
    if (match) {
      // Find the line containing the match for context
      const lines = fileContent.substring(0, match.index).split('\n');
      const line = lines[lines.length - 1] + fileContent.substring(match.index).split('\n')[0];
      const lineNumber = lines.length;
      console.log(`Found "${match[0]}" on line ${lineNumber}: "${line.trim()}"`);

      const adjustedReplacement = adjustCase(match[0], replaceValue);

      rl.question(`Replace this instance of "${match[0]}" with "${adjustedReplacement}" in ${filePath}? (Y/n): `, (answer) => {
        if (answer.toLowerCase() === 'y' || answer === '') {
          // Replace only this specific match
          fileContent = fileContent.substring(0, match.index) + adjustedReplacement + fileContent.substring(match.index + match[0].length);
          // Adjust the regex's lastIndex since the content length has changed
          regex.lastIndex = match.index + adjustedReplacement.length;
        }
        replaceNextMatch(); // Check for the next match
      });
    } else {
      // No more matches, write the file and callback
      fs.writeFile(filePath, fileContent, 'utf8', (err) => {
        if (err) console.error(`Error writing ${filePath}: ${err}`);
        else console.log(`Finished processing ${filePath}`);
        callback();
      });
    }
  };

  replaceNextMatch();
};

const searchFiles = (startPath, pattern, replaceValue, callback) => {
  fs.readdir(startPath, (err, files) => {
    if (err) {
      console.error(`Error reading directory ${startPath}: ${err}`);
      return callback();
    }

    let index = 0;
    const processNextFile = () => {
      if (index >= files.length) {
        return callback(); // All files processed
      }

      const file = files[index++];
      if (ignoreFileNames.includes(file)) {
        return processNextFile(); // Skip ignored files or directories
      }
      const filePath = path.join(startPath, file);

      fs.stat(filePath, (err, stat) => {
        if (err) {
          console.error(`Error reading file stats ${filePath}: ${err}`);
          return processNextFile(); // Skip this file
        }

        if (stat.isDirectory()) {
          searchFiles(filePath, pattern, replaceValue, processNextFile); // Recurse into subdirectories
        } else {
          fs.readFile(filePath, 'utf8', (err, fileContent) => {
            if (err) {
              console.error(`Error reading ${filePath}: ${err}`);
              return processNextFile(); // Skip this file
            }

            if (fileContent.match(new RegExp(pattern, 'i'))) { // Case-insensitive search
              confirmAndReplaceInFile(filePath, pattern, replaceValue, fileContent, processNextFile);
            } else {
              processNextFile(); // No match found, move to next file
            }
          });
        }
      });
    };

    processNextFile();
  });
};

rl.question(`Enter the pattern to replace (default: ${defaultPattern}): `, (pattern) => {
  const searchValue = pattern || defaultPattern;
  rl.question("Enter the new name (leave blank to use adjusted default based on found pattern): ", (replaceValue) => {
      const finalReplaceValue = replaceValue || defaultReplacement; // Use the user-specified value or default if blank
      searchFiles(searchDirectory, searchValue, finalReplaceValue, () => {
        console.log('All files processed.');
        rl.close();
      });
  });
});

