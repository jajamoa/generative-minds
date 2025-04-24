import React, { useState, useEffect } from 'react';
import { Box, Button, IconButton, Paper, TextField, Dialog, DialogContent, DialogActions, DialogTitle } from '@mui/material';
import { ChevronLeft, ChevronRight, Menu as MenuIcon, Close, Code as CodeIcon } from '@mui/icons-material';
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
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [codeDialogOpen, setCodeDialogOpen] = useState(false);
  const [mmdCode, setMmdCode] = useState('');
  const [layout, setLayout] = useState<'default' | 'force' | 'tree'>('default');

  // Check if the device is mobile
  const [isMobile, setIsMobile] = useState(false);
  
  useEffect(() => {
    const checkIfMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    
    // Initial check
    checkIfMobile();
    
    // Add event listener for window resize
    window.addEventListener('resize', checkIfMobile);
    
    // Cleanup
    return () => window.removeEventListener('resize', checkIfMobile);
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    const mmdFiles = files
      .filter(file => file.name.endsWith('.mmd'))
      .sort((a, b) => a.name.localeCompare(b.name));
    
    const newGraphs = await Promise.all(
      mmdFiles.map(async (file) => {
        const content = await readFileAsText(file);
        const parsedGraph = await parseMermaidFile(content);
        return {
          ...parsedGraph,
          fileName: file.name
        };
      })
    );
    
    // Keep existing graphs and add new ones
    setGraphs(prev => [...prev, ...newGraphs]);
    setCurrentIndex(graphs.length); // Set index to the first new graph
    event.target.value = ''; // Reset input to allow re-upload of the same files
  };

  const handleSingleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.name.endsWith('.mmd')) {
      const content = await readFileAsText(file);
      const parsedGraph = await parseMermaidFile(content);
      const newGraph = {
        ...parsedGraph,
        fileName: file.name
      };
      // Keep existing graphs and add new one
      setGraphs(prev => [...prev, newGraph]);
      setCurrentIndex(graphs.length); // Set index to the new graph
      event.target.value = ''; // Reset input to allow re-upload of the same file
    }
  };

  const handlePrevious = () => {
    setCurrentIndex(prev => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    setCurrentIndex(prev => Math.min(graphs.length - 1, prev + 1));
  };

  // Delete a graph from the list
  const handleDeleteGraph = (index: number) => {
    setGraphs(prev => {
      const newGraphs = [...prev];
      newGraphs.splice(index, 1);
      return newGraphs;
    });
    
    // Adjust current index if needed
    if (graphs.length <= 1) {
      setCurrentIndex(0);
    } else if (index <= currentIndex) {
      setCurrentIndex(prev => Math.max(0, prev - 1));
    }
  };

  // Move a graph up in the list
  const handleMoveUp = (index: number) => {
    if (index > 0) {
      setGraphs(prev => {
        const newGraphs = [...prev];
        [newGraphs[index - 1], newGraphs[index]] = [newGraphs[index], newGraphs[index - 1]];
        return newGraphs;
      });
      
      // Adjust current index if needed
      if (index === currentIndex) {
        setCurrentIndex(prev => prev - 1);
      } else if (index - 1 === currentIndex) {
        setCurrentIndex(prev => prev + 1);
      }
    }
  };

  // Move a graph down in the list
  const handleMoveDown = (index: number) => {
    if (index < graphs.length - 1) {
      setGraphs(prev => {
        const newGraphs = [...prev];
        [newGraphs[index], newGraphs[index + 1]] = [newGraphs[index + 1], newGraphs[index]];
        return newGraphs;
      });
      
      // Adjust current index if needed
      if (index === currentIndex) {
        setCurrentIndex(prev => prev + 1);
      } else if (index + 1 === currentIndex) {
        setCurrentIndex(prev => prev - 1);
      }
    }
  };

  // Select a graph to display
  const handleSelectGraph = (index: number) => {
    setCurrentIndex(index);
    setSidebarOpen(false);
  };

  // Toggle sidebar visibility
  const toggleSidebar = () => {
    setSidebarOpen(prev => !prev);
  };

  // Handler for MMD code paste
  const handleCodeSubmit = async () => {
    if (mmdCode.trim()) {
      // Preprocess mermaid code, convert graph TD format
      const processedCode = preprocessMermaidCode(mmdCode);
      
      const parsedGraph = await parseMermaidFile(processedCode);
      const newGraph = {
        ...parsedGraph,
        fileName: `paste-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.mmd`
      };
      // Keep existing graphs and add new one
      setGraphs(prev => [...prev, newGraph]);
      setCurrentIndex(graphs.length); // Set index to the new graph
      setCodeDialogOpen(false);
      setMmdCode(''); // Clear the text area
    }
  };

  // Preprocess mermaid code to support graph TD format
  const preprocessMermaidCode = (code: string): string => {
    const lines = code.split('\n');
    const convertedLines = [];
    
    // Check if it's graph TD format
    const isGraphTD = lines.some(line => /^\s*graph\s+TD\s*;?/.test(line));
    
    if (!isGraphTD) {
      return code; // If not graph TD format, return original code
    }
    
    // Flag to track if inside a code block
    let isInCodeBlock = false;
    
    // Preserve title
    const titleLine = lines.find(line => line.includes('title['));
    if (titleLine) {
      convertedLines.push(titleLine);
    }
    
    // Process classDef declarations
    const classDefPattern = /\s*classDef\s+(\w+)\s+(.*?);?$/;
    const classDefLines = lines.filter(line => line.match(classDefPattern));
    
    // Node and edge mappings
    const nodes: Record<string, {id: string, label: string, className?: string}> = {};
    const edges: {source: string, target: string, label?: string}[] = [];
    
    // Step 1: Process node definitions, handling multi-line labels
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      
      // Skip code block markers and graph declaration
      if (line.includes('```')) {
        isInCodeBlock = !isInCodeBlock;
        i++;
        continue;
      }
      if (line.match(/^\s*graph\s+TD\s*;?/) || line.includes('title[') || line.match(classDefPattern)) {
        i++;
        continue;
      }
      
      // Match node definition start (may be multi-line)
      // Support any alphanumeric ID, not just n\d+
      const nodeStartMatch = line.match(/\s*([A-Za-z0-9_-]+)\s*\[\"?(.*?)(?:\"?\](?:::(\w+))?)?$/);
      
      if (nodeStartMatch && !line.includes(']')) {
        // This is a multi-line node definition
        let fullNodeText = line;
        let j = i + 1;
        
        // Continue reading lines until we find the closing bracket
        while (j < lines.length && !lines[j].includes(']')) {
          fullNodeText += '\n' + lines[j];
          j++;
        }
        
        // Add the line with the closing bracket
        if (j < lines.length) {
          fullNodeText += '\n' + lines[j];
          i = j; // Update outer loop index
        }
        
        // Try to match the complete multi-line node definition
        const multiLineMatch = fullNodeText.match(/\s*([A-Za-z0-9_-]+)\s*\[\"?([\s\S]*?)\"?\](?:::(\w+))?/);
        if (multiLineMatch) {
          const [, id, rawLabel, className] = multiLineMatch;
          
          // Clean up label and preserve line breaks
          const cleanedLabel = rawLabel.split('\n')
            .map(line => line.trim())
            .join('\n');
          
          nodes[id] = { id, label: cleanedLabel, className };
        }
      } else if (nodeStartMatch && line.includes(']')) {
        // Single-line node definition
        const [, id, label, className] = nodeStartMatch;
        nodes[id] = { id, label, className };
      }
      
      i++;
    }
    
    // Step 2: Parse edges
    lines.forEach(line => {
      if (line.includes('```') || line.match(/^\s*graph\s+TD\s*;?/) || 
          line.includes('title[') || line.match(classDefPattern)) {
        return;
      }
      
      // Match edge definitions with labels - support any alphanumeric ID
      const edgeMatch = line.match(/\s*([A-Za-z0-9_-]+)\s*(==>|-->)\s*(?:\|(.*?)\|)?\s*([A-Za-z0-9_-]+)/);
      if (edgeMatch) {
        const [, source, , label, target] = edgeMatch;
        edges.push({ source, target, label });
      }
    });
    
    // Generate new Mermaid code
    convertedLines.push('');
    
    // Output node definitions
    Object.values(nodes).forEach(node => {
      const classAttr = node.className ? `:::${node.className}` : '';
      convertedLines.push(`${node.id}["${node.label}"]${classAttr}`);
    });
    
    // Output edge definitions, using --> format
    edges.forEach(edge => {
      const labelPart = edge.label ? `|${edge.label}|` : '';
      convertedLines.push(`${edge.source} --> ${labelPart}${edge.target}`);
    });
    
    // Add all classDef definitions
    classDefLines.forEach(line => {
      convertedLines.push(line);
    });
    
    return convertedLines.join('\n');
  };

  const handleChangeLayout = (newLayout: 'default' | 'force' | 'tree') => {
    setLayout(newLayout);
  };

  return (
    <Box sx={{ 
      height: '100vh', 
      display: 'flex', 
      flexDirection: 'column',
      bgcolor: '#ffffff',
      p: 2,
      gap: 1,
      position: 'relative',
    }}>
      {isMobile && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: '#ffffff',
          zIndex: 9999,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          padding: '20px',
          textAlign: 'center',
          fontFamily: 'Helvetica Neue',
        }}>
          <img 
            src="/logo512.png" 
            alt="MMDit Logo" 
            style={{
              width: '80px',
              height: '80px',
              marginBottom: '32px',
            }}
          />
          <div style={{ 
            fontSize: '20px', 
            fontWeight: '500', 
            marginBottom: '12px',
            color: '#000000',
            letterSpacing: '-0.2px',
          }}>
            Desktop Version Required
          </div>
          <div style={{ 
            fontSize: '14px', 
            maxWidth: '280px', 
            lineHeight: '1.4',
            color: '#666666',
            letterSpacing: '-0.1px',
          }}>
            This application is designed for desktop use. Please open on a larger screen for the best experience.
          </div>
        </div>
      )}
    
      <Paper sx={{ 
        p: 1.5,
        display: 'flex', 
        alignItems: 'center', 
        gap: 2, 
        bgcolor: '#ffffff',
        flexShrink: 0,
        borderRadius: '4px',
      }}>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button 
            variant="outlined" 
            component="label"
            sx={{
              borderColor: '#000000',
              color: '#000000',
              borderRadius: '4px',
              '&:hover': {
                borderColor: '#000000',
                backgroundColor: '#f8f8f8',
              }
            }}
          >
            Upload File
            <input
              type="file"
              hidden
              accept=".mmd"
              onChange={handleSingleFileUpload}
            />
          </Button>
          <Button 
            variant="outlined" 
            component="label"
            sx={{
              borderColor: '#000000',
              color: '#000000',
              borderRadius: '4px',
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
          <Button
            variant="outlined"
            onClick={() => setCodeDialogOpen(true)}
            sx={{
              borderColor: '#000000',
              color: '#000000',
              borderRadius: '4px',
              '&:hover': {
                borderColor: '#000000',
                backgroundColor: '#f8f8f8',
              }
            }}
            startIcon={<CodeIcon />}
          >
            Paste MMD
          </Button>
        </Box>
        
        {graphs.length > 0 ? (
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
              <div>{`${currentIndex + 1} / ${graphs.length}`}</div>
              <div style={{ fontSize: 12, color: '#666' }}>{graphs[currentIndex].fileName}</div>
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
        ) : (
          <Box sx={{ flex: 1 }} />
        )}
        
        <IconButton 
          onClick={toggleSidebar}
          sx={{ 
            color: '#000000',
            padding: 1,
          }}
        >
          <MenuIcon />
        </IconButton>
      </Paper>
      
      {/* Layout Controls Panel */}
      {graphs.length > 0 && (
        <div style={{
          position: 'absolute',
          bottom: '40px',
          right: '24px',
          zIndex: 5,
          backgroundColor: '#ffffff',
          border: '1px solid #000000',
          borderRadius: '4px',
          padding: '0',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 3px 10px rgba(0,0,0,0.12)',
          width: '140px',
        }}>
          <div style={{ 
            padding: '10px 16px', 
            borderBottom: '1px solid #e0e0e0',
            fontSize: '14px',
            fontWeight: 500,
            color: '#000000',
            fontFamily: 'Helvetica Neue',
            backgroundColor: '#fafafa',
          }}>
            Layout
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <button 
              onClick={() => handleChangeLayout('default')}
              style={{
                background: layout === 'default' ? '#f2f2f2' : 'none',
                border: 'none',
                textAlign: 'left',
                padding: '8px 16px',
                fontSize: '14px',
                fontFamily: 'Helvetica Neue',
                cursor: 'pointer',
                color: '#000',
                borderBottom: '1px solid #f1f1f1',
                height: '36px',
                width: '100%',
                boxSizing: 'border-box',
              }}
            >
              <span style={{ 
                fontWeight: layout === 'default' ? 500 : 400,
                display: 'inline-block',
                width: '100%',
                position: 'relative',
                paddingLeft: layout === 'default' ? '0' : '1px',  // 微调，防止选中时的轻微位移
              }}>
                Default
              </span>
            </button>
            <button 
              onClick={() => handleChangeLayout('force')}
              style={{
                background: layout === 'force' ? '#f2f2f2' : 'none',
                border: 'none',
                textAlign: 'left',
                padding: '8px 16px',
                fontSize: '14px',
                fontFamily: 'Helvetica Neue',
                cursor: 'pointer',
                color: '#000',
                borderBottom: '1px solid #f1f1f1',
                height: '36px',
                width: '100%',
                boxSizing: 'border-box',
              }}
            >
              <span style={{ 
                fontWeight: layout === 'force' ? 500 : 400,
                display: 'inline-block',
                width: '100%',
                position: 'relative',
                paddingLeft: layout === 'force' ? '0' : '1px',
              }}>
                Force-directed
              </span>
            </button>
            <button 
              onClick={() => handleChangeLayout('tree')}
              style={{
                background: layout === 'tree' ? '#f2f2f2' : 'none',
                border: 'none',
                textAlign: 'left',
                padding: '8px 16px',
                fontSize: '14px',
                fontFamily: 'Helvetica Neue',
                cursor: 'pointer',
                color: '#000',
                height: '36px',
                width: '100%',
                boxSizing: 'border-box',
              }}
            >
              <span style={{ 
                fontWeight: layout === 'tree' ? 500 : 400,
                display: 'inline-block',
                width: '100%',
                position: 'relative',
                paddingLeft: layout === 'tree' ? '0' : '1px',
              }}>
                Tree view
              </span>
            </button>
          </div>
        </div>
      )}
      
      {/* Sidebar for file management */}
      {sidebarOpen && (
        <div
          style={{
            position: 'absolute',
            top: '70px',
            right: '24px',
            maxHeight: 'calc(100vh - 140px)',
            width: '280px',
            backgroundColor: '#ffffff',
            borderRadius: '4px',
            zIndex: 10,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            boxShadow: '0 3px 10px rgba(0,0,0,0.12)',
            border: '1px solid #e0e0e0',
          }}
        >
          <div
            style={{
              padding: '14px 16px',
              borderBottom: '1px solid #e0e0e0',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              backgroundColor: '#fafafa',
            }}
          >
            <h3 style={{ 
              margin: 0, 
              fontFamily: 'Helvetica Neue', 
              fontSize: '14px', 
              fontWeight: 500,
              color: '#000000'
            }}>Uploaded Files</h3>
            <IconButton 
              onClick={toggleSidebar}
              style={{
                padding: '4px',
                color: '#000000',
              }}
            >
              <Close fontSize="small" />
            </IconButton>
          </div>
          
          <div
            style={{
              overflowY: 'auto',
              flex: 1,
              maxHeight: 'calc(100vh - 240px)',
              padding: '4px 0',
            }}
          >
            {graphs.length === 0 ? (
              <div style={{
                padding: '20px',
                textAlign: 'center',
                color: '#666',
                fontSize: '14px',
                fontFamily: 'Helvetica Neue'
              }}>
                No files uploaded yet
              </div>
            ) : (
              graphs.map((graph, index) => (
                <div
                  key={index}
                  style={{
                    padding: '10px 16px',
                    display: 'flex',
                    alignItems: 'center',
                    backgroundColor: index === currentIndex ? '#f2f2f2' : 'transparent',
                    cursor: 'pointer',
                    transition: 'background-color 0.15s ease',
                    margin: '0 0',
                    borderBottom: '1px solid #f1f1f1',
                  }}
                  onMouseEnter={(e) => {
                    if (index !== currentIndex) {
                      e.currentTarget.style.backgroundColor = '#f8f8f8';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (index !== currentIndex) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }
                  }}
                >
                  <div
                    style={{
                      flex: 1,
                      fontFamily: 'Helvetica Neue',
                      fontSize: '14px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      color: index === currentIndex ? '#000000' : '#333333',
                      fontWeight: index === currentIndex ? 500 : 400,
                    }}
                    onClick={() => handleSelectGraph(index)}
                  >
                    {graph.fileName}
                  </div>
                  <div 
                    style={{ 
                      display: 'flex', 
                      gap: '0px',
                    }}
                  >
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleMoveUp(index);
                      }}
                      disabled={index === 0}
                      style={{
                        border: 'none',
                        background: 'none',
                        cursor: index === 0 ? 'default' : 'pointer',
                        opacity: index === 0 ? 0.3 : 1,
                        padding: '4px 8px',
                        fontSize: '14px',
                        color: '#000',
                        fontWeight: 'bold',
                      }}
                    >
                      ↑
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleMoveDown(index);
                      }}
                      disabled={index === graphs.length - 1}
                      style={{
                        border: 'none',
                        background: 'none',
                        cursor: index === graphs.length - 1 ? 'default' : 'pointer',
                        opacity: index === graphs.length - 1 ? 0.3 : 1,
                        padding: '4px 8px',
                        fontSize: '14px',
                        color: '#000',
                        fontWeight: 'bold',
                      }}
                    >
                      ↓
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteGraph(index);
                      }}
                      style={{
                        border: 'none',
                        background: 'none',
                        cursor: 'pointer',
                        padding: '4px 8px',
                        fontSize: '14px',
                        color: '#000',
                        fontWeight: 'bold',
                      }}
                    >
                      ×
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
          
          <div style={{
            padding: '10px 16px',
            borderTop: '1px solid #e0e0e0',
            display: 'flex',
            justifyContent: 'space-between',
            color: '#666',
            fontSize: '13px',
            fontFamily: 'Helvetica Neue',
            backgroundColor: '#fafafa',
          }}>
            <span>{graphs.length} file(s)</span>
            {graphs.length > 0 && (
              <span>Current: {currentIndex + 1}</span>
            )}
          </div>
        </div>
      )}
      
      {/* MMD Code Paste Dialog */}
      <Dialog 
        open={codeDialogOpen} 
        onClose={() => setCodeDialogOpen(false)}
        fullWidth
        maxWidth="md"
        PaperProps={{
          style: {
            borderRadius: '4px',
            boxShadow: '0 3px 10px rgba(0,0,0,0.12)',
            border: '1px solid #e0e0e0',
          }
        }}
      >
        <DialogTitle style={{ 
          borderBottom: '1px solid #e0e0e0',
          padding: '14px 16px',
          fontFamily: 'Helvetica Neue',
          fontSize: '14px',
          fontWeight: 500,
          backgroundColor: '#fafafa',
          color: '#000000',
          margin: 0,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span>Paste Mermaid Markdown Code</span>
          <IconButton 
            onClick={() => setCodeDialogOpen(false)}
            style={{
              padding: '4px',
              color: '#000000',
            }}
          >
            <Close fontSize="small" />
          </IconButton>
        </DialogTitle>
        <DialogContent style={{ padding: '16px' }}>
          <TextField
            autoFocus
            multiline
            rows={15}
            variant="outlined"
            fullWidth
            value={mmdCode}
            onChange={(e) => setMmdCode(e.target.value)}
            placeholder="Paste your MMD code here..."
            InputProps={{
              style: {
                fontFamily: 'monospace',
                fontSize: '14px',
                borderRadius: '4px',
              }
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': {
                  borderColor: '#000000',
                  borderRadius: '4px',
                  borderWidth: '1px',
                },
                '&:hover fieldset': {
                  borderColor: '#000000',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#000000',
                },
              },
            }}
          />
        </DialogContent>
        <DialogActions style={{ 
          padding: '10px 16px',
          borderTop: '1px solid #e0e0e0',
          backgroundColor: '#fafafa',
          justifyContent: 'flex-end',
        }}>
          <Button 
            onClick={handleCodeSubmit}
            variant="outlined"
            disabled={!mmdCode.trim()}
            sx={{
              borderColor: '#000000',
              color: '#000000',
              borderRadius: '4px',
              fontFamily: 'Helvetica Neue',
              textTransform: 'none',
              fontSize: '14px',
              padding: '6px 16px',
              minWidth: '80px',
              fontWeight: 400,
              '&:hover': {
                borderColor: '#000000',
                backgroundColor: '#f8f8f8',
              },
              '&.Mui-disabled': {
                borderColor: '#cccccc',
                color: '#cccccc',
              }
            }}
          >
            Visualize
          </Button>
        </DialogActions>
      </Dialog>
      
      {graphs.length > 0 ? (
        <Paper sx={{ 
          flex: 1, 
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
          borderRadius: '4px',
        }}>
          <NetworkGraph
            nodes={graphs[currentIndex].nodes}
            edges={graphs[currentIndex].edges}
            layout={layout}
          />
        </Paper>
      ) : (
        <Paper sx={{ 
          flex: 1, 
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          color: '#666',
          fontFamily: 'Helvetica Neue',
          gap: 2,
          borderRadius: '4px',
        }}>
          <div style={{ fontSize: 16 }}>No graphs uploaded</div>
          <div style={{ fontSize: 14 }}>Please upload a .mmd file or folder to visualize</div>
        </Paper>
      )}
    </Box>
  );
}

export default App;
