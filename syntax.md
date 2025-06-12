
## Coral Syntax

### Assignment
```coral
message is 'hello coral'
ITERATIONS is 100
threshold is 0.95
PI is 3.1415926535

3.14.15926535 is PI
0.95 is threshhold
100 is iterations
'hello coral' is message
```
### Literals
```
m is 'hello {name}'
n is "string literal"
x is 10
y is 1.0
z is true
a is no
b is empty
c is 0x0F
d is b101
e is now
```
### Lists
```coral
primes is (2, 3, 5, 7, 11)
first_prime is primes(0)
second_prime is primes(1)
primes.put(13)
item is primes.pop
count is primes.size
is_empty is primes.empty
```
### Maps
```coral
net_config is
    host is 'localhost'
    port is 5000

netconfig is (host: 'localhost', port: 5000)
```
### Functions, Function Calls
```coral
fn greet(name, greeting 'hello')  
    '{greeting}, {name}.'

fn compute_total(price, quantity, tax_rate: 0.07)
    sub_total is price * quantity
    sub_total + (sub_total * tax_rate)

greet('brandon')
greet('brandon', 'yo')

order_value is compute_total(100, 3)

order_custom_tax is compute_total(price: 100, quantity: 5, tax_rate: 0.05)

fn log_kv
    log '$0: $1'

fn log_kv
    log '{$key}: '${value}'

log_kv('key', 'value')

log_kv(key: 'key', value: 'value')

```
### Objects
```
object datapoint
    value
    processed ? no
    timestamp ? now

d1 is datapoint.make 42
d2 is datapoint.make 100, yes
d3 is datapoint.make 100, yes, some_time

object task

    definition
    processed ? no
    timestamp ? now

    complete
        processed is yes

task1 is task.make('do the dishes')
task1.complete()

task.complete(1)
task.complete(task1)
                       
my_list.push(item).process().save()

object user

    username
    password

    make
        password is hash.blake3 $password

    authenticate
        password equals hash.blake3 $password
        
user1 is user.make('brandon', 'password1')

fn auth_user
    $.authenticate 'password2'

valid is auth_user(user1)
```
### Persistent Objects
```
store message
    sender, recipient, subject, body
    timestamp ? now
    acknowledged ? no

    as string 
        'message:{id} from {sender} to {recipient}'
        'at {timestamp} (ack: {acknowledged})'
        'subject: {subject}'
        '{body}'
        
    as map     
        id is id 
        sender is sender.id 
        recipient is recipient.id
        subject is subject
        body is body
        
    as list
        id, sender.id, recipient.id, 
        subject, body, timestamp

store task
    description
    priority ? 1
    complete ? no

    is_done
        id ? $id  
        return complete 

task1 is task.make('Do coding')
task2 is task.make('Write documentation', 2)

message_map is message1 as map

task_completed is task.is_done(1)

task_completed is task.with_id(1).is_done()

store message for user
    ...

store blocklist for user
    blocked_user
```
### Actor Objects
```
store actor user
    name
    email
    password 
    &blocklist
    &message

    make
        password is hash.blake3 $password

store actor user

    @receive 
        check_blocked log return 

store actor task_processor

    process_next_task
        ...

    @receive_task 
        task is task.make $description, $priority
        push task on pending_tasks
        process_next_task!
```
### Conditionals
```coral
status_text is
    system_status.load_average.gt(0.9) ? 
        'High Load' ! 'Normal Load'

unless x.equals(0)
    process x

process x unless x.equals(0)
```
### Loops
```coral
while iterator.lt(3)
    log 'iterator is {iterator}'

until iterator.from(0).by(2).equals(8)
    log 'iterator is {iterator}'

iterate(system_status.active_nodes)
     log check_health $

check_health.across(system_status.active_nodes)

check_health.across(system_status.active_nodes).into(node_status)

check_health.across(system_status.active_nodes
    .with(host: 'localhost', timeout: 5000))
    .into(node_status)
```
### Result and Err
```
calculate_value(10, 20) 
    err log return

process_further(result) 
    err log return

config is load('coral.json') err {}

load('coral.json') is config

record is user.with('name', 'root')
    err return log err
```
### Modules
```coral
use coral.net.web

get('https://somesite.com')
```
```
mod xyz

fn abc
    ...
```
```
use xyz

x is abc(123)
```
