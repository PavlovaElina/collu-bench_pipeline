#lang racket
(define (digit-sum n)
  (define digs (map (lambda (c) (- (char->integer c) 48))
                    (string->list (number->string (abs n)))))
  (if (< n 0)
      (apply + (cons (- (car digs)) (cdr digs)))
      (apply + digs)))
(define (count_nums lst)
  (length (filter (lambda (n) (> (digit-sum n) 0)) lst)))
