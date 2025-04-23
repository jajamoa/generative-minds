import React, { useState } from 'react';
import { Box, Button, IconButton, Paper } from '@mui/material';
import { ChevronLeft, ChevronRight } from '@mui/icons-material';
import NetworkGraph from './components/NetworkGraph';
import { parseMermaidFile, readFileAsText } from './utils/mermaidParser';
import { Node, Edge } from 'reactflow';
import 'reactflow/dist/style.css';

// 扩展HTMLInputElement的类型定义
declare module 'react' {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    directory?: string;
    webkitdirectory?: string;
  }
}

interface GraphData {
  nodes: Node[];
  edges: Edge[];
  fileName: string; // 添加文件名字段
}

function App() {
  const [graphs, setGraphs] = useState<GraphData[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    const mmdFiles = files
      .filter(file => file.name.endsWith('.mmd'))
      .sort((a, b) => a.name.localeCompare(b.name)); // 按文件名排序
    
    const newGraphs = await Promise.all(
      mmdFiles.map(async (file) => {
        const content = await readFileAsText(file);
        return {
          ...parseMermaidFile(content),
          fileName: file.name
        };
      })
    );
    
    setGraphs(newGraphs);
    setCurrentIndex(0);
  };

  const handlePrevious = () => {
    setCurrentIndex(prev => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    setCurrentIndex(prev => Math.min(graphs.length - 1, prev + 1));
  };

  return (
    <Box sx={{ 
      height: '100vh', 
      display: 'flex', 
      flexDirection: 'column',
      bgcolor: '#ffffff',
      p: 2,
      gap: 1,
    }}>
      <Paper sx={{ 
        p: 1.5,
        display: 'flex', 
        alignItems: 'center', 
        gap: 2, 
        bgcolor: '#ffffff',
        flexShrink: 0,
      }}>
        <Button 
          variant="outlined" 
          component="label"
          sx={{
            borderColor: '#000000',
            color: '#000000',
            '&:hover': {
              borderColor: '#000000',
              backgroundColor: '#f8f8f8',
            }
          }}
        >
          Upload Folder
          <input
            type="file"
            hidden
            multiple
            directory=""
            webkitdirectory=""
            onChange={handleFileUpload}
          />
        </Button>
        <Box sx={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 2 }}>
          <IconButton 
            onClick={handlePrevious} 
            disabled={currentIndex === 0}
            sx={{ 
              color: '#000000',
              '&.Mui-disabled': {
                color: '#cccccc'
              }
            }}
          >
            <ChevronLeft />
          </IconButton>
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            fontFamily: 'Helvetica Neue', 
            fontSize: 14, 
            letterSpacing: 0.3 
          }}>
            {graphs.length > 0 ? (
              <>
                <div>{`${currentIndex + 1} / ${graphs.length}`}</div>
                <div style={{ fontSize: 12, color: '#666' }}>{graphs[currentIndex].fileName}</div>
              </>
            ) : (
              'No graphs'
            )}
          </Box>
          <IconButton 
            onClick={handleNext} 
            disabled={currentIndex === graphs.length - 1}
            sx={{ 
              color: '#000000',
              '&.Mui-disabled': {
                color: '#cccccc'
              }
            }}
          >
            <ChevronRight />
          </IconButton>
        </Box>
      </Paper>
      
      {graphs.length > 0 && (
        <Paper sx={{ 
          flex: 1, 
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
        }}>
          <NetworkGraph
            nodes={graphs[currentIndex].nodes}
            edges={graphs[currentIndex].edges}
          />
        </Paper>
      )}
    </Box>
  );
}

export default App;
