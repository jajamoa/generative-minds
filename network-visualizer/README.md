# MMDit - Mermaid Markdown Visualizer

MMDit is an elegant, interactive visualization tool for Mermaid Markdown network graphs. It allows users to upload, visualize, and manipulate network diagrams with multiple layout options.

## Features

- **Multiple Upload Methods**: Upload single .mmd files, entire folders, or paste Mermaid code directly
- **Flexible Layout Options**: Choose between default, force-directed, and tree view layouts
- **File Management**: Save and organize multiple graphs with an intuitive file management sidebar
- **Interaction**: Pan, zoom, and drag nodes to explore complex network relationships
- **Clean UI**: Minimalist black and white interface focused on your data visualization

## Getting Started

### Installation

```bash
git clone https://github.com/your-username/mmdit.git
cd mmdit
npm install
```

### Development

```bash
npm start
```

This runs the app in development mode. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### Production Build

```bash
npm run build
```

Builds the app for production to the `build` folder, optimized for performance.

## Usage

1. **Upload Files**: Use the "Upload File" or "Upload Folder" buttons to load Mermaid Markdown files
2. **Paste MMD**: Click the "Paste MMD" button to directly enter Mermaid code
3. **Navigate**: Use the navigation arrows to move between multiple loaded graphs
4. **Manage Files**: Click the menu icon to open the file management sidebar
5. **Change Layout**: Select different layout algorithms from the layout panel

## Technical Details

Built with:
- React
- TypeScript
- ReactFlow
- Material UI components

## License

MIT License

## Developed By

Generative Minds Lab, MIT

---

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).
