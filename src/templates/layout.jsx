import React from 'react';
import Layout from '@theme/Layout';

export default function {templateName ?? 'MyTemplate'}(props) {
  const title = props.title || '{templateName ?? 'My Template'}';
  const description = props.description || 'Default description';

  return (
    <Layout
      title={title}
      description={description}>
      <main>
        <div className="container margin-vert--lg">
          <div className="row">
            <div className="col col--8 col--offset-2">
              <h1 className="hero__title">{title}</h1>
              <p>{description}</p>
              
              {/* Add your content here */}
              {props.children}
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}