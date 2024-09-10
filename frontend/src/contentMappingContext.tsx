// contentMappingContext.tsx
import React, { createContext, useContext, useState, useEffect } from 'react';

type ContentMapping = {
  URL: string;
  FileName: string;
}[];

const ContentMappingContext = createContext<ContentMapping | null>(null);

export const useContentMapping = () => useContext(ContentMappingContext);

export const ContentMappingProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  const [contentMapping, setContentMapping] = useState<ContentMapping | null>(null);

  useEffect(() => {
    const fetchContentMapping = async () => {
      try {
        const response = await fetch('/api/content-mapping');
        if (!response.ok) {
          throw new Error('Failed to fetch content mapping');
        }
        const data = await response.json();
        setContentMapping(data);
      } catch (error) {
        console.error('Error fetching content mapping:', error);
      }
    };

    fetchContentMapping();
  }, []);

  return (
    <ContentMappingContext.Provider value={contentMapping}>
      {children}
    </ContentMappingContext.Provider>
  );
};