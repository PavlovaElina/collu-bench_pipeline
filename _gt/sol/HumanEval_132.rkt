#lang racket
(define (is_nested str)
  (define n (string-length str))
  (define opening (for/list ([i (in-range n)] #:when (char=? (string-ref str i) #\[)) i))
  (define closing (reverse (for/list ([i (in-range n)] #:when (char=? (string-ref str i) #\])) i)))
  (define cl (list->vector closing))
  (define l (vector-length cl))
  (let loop ([ops opening] [i 0] [cnt 0])
    (cond [(null? ops) (>= cnt 2)]
          [(and (< i l) (< (car ops) (vector-ref cl i))) (loop (cdr ops) (+ i 1) (+ cnt 1))]
          [else (loop (cdr ops) i cnt)])))
