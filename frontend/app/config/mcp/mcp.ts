export class McpConfig {
  id: string;
  name: string;
  url: string;
  type: string;
  auth_token: string;
  need_token: boolean;
  enabled: boolean;

  constructor(
    id: string,
    name: string,
    url: string,
    type: string,
    auth_token: string,
    need_token: boolean,
    enabled: boolean,
  ) {
    this.id = id;
    this.name = name;
    this.url = url;
    this.type = type;
    this.auth_token = auth_token;
    this.need_token = need_token;
    this.enabled = enabled;
  }
};